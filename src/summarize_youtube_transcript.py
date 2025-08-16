import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from src.utils import (
    ensure_output_folder,
    get_config,
    get_prompt_path,
    get_worker_count,
    setup_logging,
)

load_dotenv()


class YouTubeTranscriptSummarizer:
    """A class to summarize YouTube SRT transcript files using OpenAI's GPT-5 model with custom prompts."""

    def __init__(
        self, api_key: Optional[str] = None, num_workers: Optional[int] = None
    ):
        """Initialize the YouTubeTranscriptSummarizer.

        Args:
            api_key (Optional[str]): OpenAI API key. If None, uses OPENAI_API_KEY env var.
            num_workers (Optional[int]): Number of concurrent workers for parallel processing.
                If None, auto-detects based on config and CPU count.
        """
        print("[INIT] Initializing YouTubeTranscriptSummarizer...")

        # Initialize configuration and logging
        setup_logging()

        print("[INIT] Setting up OpenAI client...")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = get_config("api.openai.model", "gpt-4o-mini")
        self.timeout = get_config("api.openai.timeout", 120)

        print("[INIT] Loading prompt template...")
        self.prompt_template = self._load_prompt_template()
        print("[INIT] Determining number of workers...")
        self.num_workers = get_worker_count(num_workers)
        print(f"[INIT] Initialized with {self.num_workers} workers")

    def _load_prompt_template(self) -> str:
        """Load the prompt template from YAML file.

        Returns:
            str: The loaded prompt template.
        """
        prompt_path = get_prompt_path()
        print(f"[PROMPT] Loading prompt template from: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt_data = yaml.safe_load(file)
            print("[PROMPT] Prompt template loaded successfully")
            return prompt_data["prompt"]

    def _read_srt_file(self, srt_path: str) -> str:
        """Read and parse SRT file content.

        Args:
            srt_path (str): Path to the SRT file.

        Returns:
            str: The content of the SRT file as plain text.
        """
        print(f"[SRT] Reading SRT file: {srt_path}")
        with open(srt_path, "r", encoding="utf-8") as file:
            content = file.read()

        print("[SRT] Parsing SRT content...")
        # Basic SRT parsing - extract just the text content
        lines = content.split("\n")
        text_lines = []

        for line in lines:
            line = line.strip()
            # Skip sequence numbers, timestamps, and empty lines
            if line and not line.isdigit() and "-->" not in line:
                text_lines.append(line)

        transcript_text = " ".join(text_lines)
        print(
            f"[SRT] Extracted {len(text_lines)} text lines, {len(transcript_text)} characters total"
        )
        return transcript_text

    def summarize_transcript(
        self,
        srt_path: str,
        video_title: str = "",
        video_description: str = "",
        output_path: Optional[str] = None,
    ) -> str:
        """Summarize a YouTube SRT transcript file using the YouTube summarizer prompt.

        Args:
            srt_path (str): Path to the SRT file to summarize.
            video_title (str): Title of the video (optional).
            video_description (str): Description of the video (optional).
            output_path (Optional[str]): Path to save the summary. If None, prints to stdout.

        Returns:
            str: The generated summary.
        """
        start_time = time.time()
        thread_id = f"T{id(threading.current_thread()) % 10000}"
        print(
            f"[SUMMARIZE-{thread_id}] Starting summarization for: {os.path.basename(srt_path)}"
        )

        # Read the SRT file
        read_start = time.time()
        transcript_text = self._read_srt_file(srt_path)
        read_time = time.time() - read_start
        print(f"[SUMMARIZE-{thread_id}] File read took {read_time:.2f}s")

        print(f"[SUMMARIZE-{thread_id}] Preparing API request message...")
        # Prepare the user message with video details and transcript
        user_message = f"""
**YouTube Video Title:** {video_title if video_title else "Not provided"}

**YouTube Video Description:** {video_description if video_description else "Not provided"}

**Full Transcript:**
{transcript_text}
"""

        try:
            api_start = time.time()
            print(f"[API-{thread_id}] Making API call to OpenAI...")

            # Create a new client instance for thread safety
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Make API call to OpenAI
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt_template},
                    {"role": "user", "content": user_message},
                ],
                timeout=self.timeout,
            )

            api_time = time.time() - api_start
            print(f"[API-{thread_id}] Received response in {api_time:.2f}s")
            summary = response.choices[0].message.content

            # Save or print the summary
            if output_path:
                save_start = time.time()
                print(
                    f"[OUTPUT-{thread_id}] Saving summary to file: {os.path.basename(output_path)}"
                )
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(summary)
                save_time = time.time() - save_start
                print(f"[OUTPUT-{thread_id}] File saved in {save_time:.2f}s")
            else:
                print(f"[OUTPUT-{thread_id}] Printing summary to stdout")
                print(summary)

            total_time = time.time() - start_time
            print(
                f"[SUMMARIZE-{thread_id}] Completed in {total_time:.2f}s (API: {api_time:.2f}s)"
            )
            return summary

        except Exception as e:
            total_time = time.time() - start_time
            print(
                f"[ERROR-{thread_id}] Error after {total_time:.2f}s for {os.path.basename(srt_path)}: {str(e)}"
            )
            return ""

    def _summarize_transcripts_sequential(
        self, transcript_paths: List[str], output_folder: Optional[str] = None
    ) -> dict:
        """Summarize transcripts sequentially (one at a time).

        Args:
            transcript_paths (List[str]): List of SRT file paths.
            output_folder (Optional[str]): Folder to save summaries. If None, prints to stdout.

        Returns:
            dict: Dictionary with file paths as keys and success status as values.
        """
        print(
            f"[SEQUENTIAL] Starting sequential processing of {len(transcript_paths)} transcripts"
        )
        results = {}
        for i, srt_path in enumerate(transcript_paths, 1):
            print(f"[SEQUENTIAL] Processing {i}/{len(transcript_paths)}: {srt_path}")
            output_path = None
            if output_folder:
                filename = (
                    os.path.splitext(os.path.basename(srt_path))[0] + "_summary.json"
                )
                output_path = os.path.join(output_folder, filename)
                print(f"[SEQUENTIAL] Output will be saved to: {output_path}")

            summary = self.summarize_transcript(srt_path, output_path=output_path)
            results[srt_path] = bool(summary)
            print(f"[SEQUENTIAL] Completed {i}/{len(transcript_paths)}")

        print("[SEQUENTIAL] All transcripts processed sequentially")
        return results

    def _summarize_transcripts_parallel(
        self, transcript_paths: List[str], output_folder: Optional[str] = None
    ) -> dict:
        """Summarize transcripts in parallel using ThreadPoolExecutor.

        Args:
            transcript_paths (List[str]): List of SRT file paths.
            output_folder (Optional[str]): Folder to save summaries. If None, prints to stdout.

        Returns:
            dict: Dictionary with file paths as keys and success status as values.
        """
        start_time = time.time()
        print(
            f"[PARALLEL] Starting parallel processing with {self.num_workers} workers for {len(transcript_paths)} files"
        )
        results = {}

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            print(
                f"[PARALLEL] ThreadPoolExecutor created with {self.num_workers} max workers"
            )
            future_to_path = {}

            # Submit all tasks at once for true parallelism
            submit_start = time.time()
            print(f"[PARALLEL] Submitting {len(transcript_paths)} tasks to executor...")
            for i, srt_path in enumerate(transcript_paths, 1):
                output_path = None
                if output_folder:
                    filename = (
                        os.path.splitext(os.path.basename(srt_path))[0]
                        + "_summary.json"
                    )
                    output_path = os.path.join(output_folder, filename)

                print(
                    f"[PARALLEL] Submitting task {i}/{len(transcript_paths)}: {os.path.basename(srt_path)}"
                )
                future = executor.submit(
                    self.summarize_transcript, srt_path, "", "", output_path
                )
                future_to_path[future] = srt_path

            submit_time = time.time() - submit_start
            print(
                f"[PARALLEL] All {len(transcript_paths)} tasks submitted in {submit_time:.2f}s, waiting for completion..."
            )

            # Track completion progress
            completed = 0
            completion_times = []
            for future in as_completed(future_to_path):
                completed += 1
                completion_time = time.time()
                completion_times.append(completion_time)

                srt_path = future_to_path[future]
                print(
                    f"[PARALLEL] Completed task {completed}/{len(transcript_paths)}: {os.path.basename(srt_path)}"
                )
                try:
                    summary = future.result()
                    success = bool(summary)
                    results[srt_path] = success
                    status = "Success" if success else "Failed"
                    elapsed = completion_time - start_time
                    print(
                        f"[PARALLEL] Task result - {os.path.basename(srt_path)}: {status} (elapsed: {elapsed:.2f}s)"
                    )
                except Exception as e:
                    results[srt_path] = False
                    print(
                        f"[PARALLEL] Task error - {os.path.basename(srt_path)}: {str(e)}"
                    )

        total_time = time.time() - start_time
        avg_time_per_task = (
            total_time / len(transcript_paths) if transcript_paths else 0
        )
        print(
            f"[PARALLEL] All parallel tasks completed in {total_time:.2f}s (avg: {avg_time_per_task:.2f}s per task)"
        )

        # Calculate if we actually got parallelism benefits
        if len(completion_times) > 1:
            first_completion = min(completion_times) - start_time
            last_completion = max(completion_times) - start_time
            print(
                f"[PARALLEL] First task completed in {first_completion:.2f}s, last in {last_completion:.2f}s"
            )
            print(
                f"[PARALLEL] Parallelism efficiency: {(avg_time_per_task * len(transcript_paths)) / total_time:.2f}x speedup"
            )

        return results

    def summarize_transcripts(
        self, transcript_paths: List[str], output_folder: Optional[str] = None
    ) -> dict:
        """Summarize multiple transcript files with automatic parallel/sequential fallback.

        Args:
            transcript_paths (List[str]): List of SRT file paths.
            output_folder (Optional[str]): Folder to save summaries. If None, prints to stdout.

        Returns:
            dict: Dictionary with file paths as keys and success status as values.
        """
        print(
            f"[BATCH] Starting batch summarization of {len(transcript_paths)} transcripts"
        )

        if not transcript_paths:
            print("[BATCH] No transcript paths provided, returning empty results")
            return {}

        if output_folder:
            output_folder = ensure_output_folder(output_folder)
            print(f"[BATCH] Using output folder: {output_folder}")
        else:
            print("[BATCH] No output folder specified, will print to stdout")

        if self.num_workers == 0 or len(transcript_paths) == 1:
            print(
                f"[BATCH] Using sequential processing for {len(transcript_paths)} transcript(s) (workers={self.num_workers})"
            )
            return self._summarize_transcripts_sequential(
                transcript_paths, output_folder
            )

        try:
            print(
                f"[BATCH] Attempting parallel processing with {self.num_workers} workers for {len(transcript_paths)} transcripts"
            )
            return self._summarize_transcripts_parallel(transcript_paths, output_folder)
        except Exception as e:
            print(
                f"[BATCH] Parallel processing failed ({str(e)}), falling back to sequential processing"
            )
            return self._summarize_transcripts_sequential(
                transcript_paths, output_folder
            )


def main():
    """Main function to run the YouTube transcript summarizer from command line."""
    print("[MAIN] Starting YouTube Transcript Summarizer")

    if len(sys.argv) < 2:
        print("[MAIN] No arguments provided, showing usage")
        print("Usage modes:")
        print(
            "  Single file: python summarize_youtube_transcript.py <srt_file_path> [video_title] [video_description] [output_path]"
        )
        print(
            "  Multiple files: python summarize_youtube_transcript.py --batch <transcript1.srt> <transcript2.srt> ... [--output-folder <folder>]"
        )
        print("Examples:")
        print(
            "  python summarize_youtube_transcript.py transcript.srt 'Video Title' 'Video Description' summary.json"
        )
        print(
            "  python summarize_youtube_transcript.py --batch file1.srt file2.srt file3.srt --output-folder summaries/"
        )
        sys.exit(1)

    if sys.argv[1] == "--batch":
        print("[MAIN] Batch mode detected")
        # Batch mode - process multiple files
        transcript_paths = []
        output_folder = None
        i = 2

        print("[MAIN] Parsing command line arguments...")
        while i < len(sys.argv):
            if sys.argv[i] == "--output-folder" and i + 1 < len(sys.argv):
                output_folder = sys.argv[i + 1]
                print(f"[MAIN] Output folder specified: {output_folder}")
                i += 2
            else:
                if os.path.exists(sys.argv[i]) and sys.argv[i].endswith(".srt"):
                    transcript_paths.append(sys.argv[i])
                    print(f"[MAIN] Added transcript file: {sys.argv[i]}")
                else:
                    print(f"[MAIN] Skipping invalid file: {sys.argv[i]}")
                i += 1

        print(f"[MAIN] Found {len(transcript_paths)} valid SRT files")
        if not transcript_paths:
            print("[MAIN] Error: No valid SRT files provided")
            sys.exit(1)

        print("[MAIN] Initializing summarizer for batch processing...")
        # Initialize summarizer and process files
        summarizer = YouTubeTranscriptSummarizer()
        print("[MAIN] Starting batch summarization...")
        results = summarizer.summarize_transcripts(transcript_paths, output_folder)

        # Print results
        print("\n[MAIN] Final Summary Results:")
        for path, success in results.items():
            status = "Success" if success else "Failed"
            print(f"[MAIN] {path}: {status}")
        print("[MAIN] Batch processing completed")

    else:
        print("[MAIN] Single file mode detected")
        # Single file mode
        srt_path = sys.argv[1]
        video_title = sys.argv[2] if len(sys.argv) > 2 else ""
        video_description = sys.argv[3] if len(sys.argv) > 3 else ""
        output_path = sys.argv[4] if len(sys.argv) > 4 else None

        print(f"[MAIN] SRT file: {srt_path}")
        print(f"[MAIN] Video title: {video_title if video_title else 'Not provided'}")
        print(
            f"[MAIN] Video description: {video_description if video_description else 'Not provided'}"
        )
        print(f"[MAIN] Output path: {output_path if output_path else 'stdout'}")

        if not os.path.exists(srt_path):
            print(f"[MAIN] Error: SRT file not found: {srt_path}")
            sys.exit(1)

        print("[MAIN] Initializing summarizer for single file processing...")
        # Initialize summarizer and process the file
        summarizer = YouTubeTranscriptSummarizer()
        print("[MAIN] Starting single file summarization...")
        result = summarizer.summarize_transcript(
            srt_path, video_title, video_description, output_path
        )

        if result:
            print("[MAIN] Single file processing completed successfully")
        else:
            print("[MAIN] Single file processing failed")


# Example usage for batch processing
if __name__ == "__main__":
    transcript_paths = [
        # "/Users/ishandutta/Documents/code/atlas/transcripts/gpz6C_2l5jI.en.srt",
        # "/Users/ishandutta/Documents/code/atlas/transcripts/q6kJ71tEYqM.en.srt",
        "/Users/ishandutta/Documents/code/atlas/transcripts/UV81LAb3x2g.en.srt"
    ]

    summarizer = YouTubeTranscriptSummarizer()
    results = summarizer.summarize_transcripts(transcript_paths, "summaries/")

    for path, success in results.items():
        status = "Success" if success else "Failed"
        print(f"{path}: {status}")
