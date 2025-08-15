#!/usr/bin/env python3
"""
YouTube Video Processing Pipeline

This script provides a complete pipeline that integrates:
1. YouTube video search using the YouTube Data API
2. Transcript fetching using yt-dlp
3. Transcript summarization using OpenAI's GPT models

The pipeline takes a search query and configuration, then automatically:
- Searches for relevant YouTube videos
- Fetches transcripts for the found videos
- Summarizes the transcripts for engineers
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from src.fetch_youtube_transcript import YouTubeTranscriptFetcher
from src.summarize_youtube_transcript import YouTubeTranscriptSummarizer
from src.utils import ensure_output_folder, get_config, setup_logging
from src.youtube_video_search import search_youtube_videos_api

load_dotenv()


class YouTubePipeline:
    """Complete pipeline for YouTube video processing.

    This class integrates video search, transcript fetching, and summarization
    into a single automated workflow.
    """

    def __init__(
        self,
        max_videos: Optional[int] = None,
        transcript_language: Optional[str] = None,
        output_folder: Optional[str] = None,
        num_workers: Optional[int] = None,
    ):
        """Initialize the YouTube pipeline.

        Args:
            max_videos (Optional[int]): Maximum number of videos to process.
                If None, uses config default.
            transcript_language (Optional[str]): Language for transcripts.
                If None, uses config default.
            output_folder (Optional[str]): Base output folder for all results.
                If None, uses current directory.
            num_workers (Optional[int]): Number of concurrent workers.
                If None, uses config default.
        """
        # Initialize configuration and logging
        setup_logging()

        # Set configuration values
        self.max_videos = max_videos or get_config("api.youtube.max_results", 5)
        self.transcript_language = transcript_language or get_config(
            "processing.transcripts.language", "en"
        )
        self.output_folder = output_folder or "pipeline_output"
        self.num_workers = num_workers

        # Ensure output directories exist
        self.output_folder = ensure_output_folder(self.output_folder)
        self.transcripts_folder = ensure_output_folder(
            os.path.join(self.output_folder, "transcripts")
        )
        self.summaries_folder = ensure_output_folder(
            os.path.join(self.output_folder, "summaries")
        )
        self.metadata_folder = ensure_output_folder(
            os.path.join(self.output_folder, "metadata")
        )

        # Initialize components
        self.transcript_fetcher = YouTubeTranscriptFetcher(
            output_folder=self.transcripts_folder,
            language=self.transcript_language,
            num_workers=self.num_workers,
        )
        self.summarizer = YouTubeTranscriptSummarizer(num_workers=self.num_workers)

        print(f"[PIPELINE] Initialized with max_videos={self.max_videos}")
        print(f"[PIPELINE] Output folder: {self.output_folder}")
        print(f"[PIPELINE] Transcripts folder: {self.transcripts_folder}")
        print(f"[PIPELINE] Summaries folder: {self.summaries_folder}")

    def search_videos(self, search_query: str) -> List[Dict]:
        """Search for YouTube videos using the query.

        Args:
            search_query (str): The search query for YouTube videos.

        Returns:
            List[Dict]: List of video information dictionaries.
        """
        print(f"[SEARCH] Searching for videos with query: '{search_query}'")
        print(f"[SEARCH] Maximum videos to retrieve: {self.max_videos}")

        try:
            videos = search_youtube_videos_api(search_query, self.max_videos)
            print(f"[SEARCH] Found {len(videos)} videos")

            # Save search results metadata
            search_metadata = {
                "search_query": search_query,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_videos_found": len(videos),
                "max_videos_requested": self.max_videos,
                "videos": videos,
            }

            metadata_path = os.path.join(
                self.metadata_folder, f"search_results_{int(time.time())}.json"
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(search_metadata, f, indent=2, ensure_ascii=False)
            print(f"[SEARCH] Search metadata saved to: {metadata_path}")

            return videos
        except Exception as e:
            print(f"[SEARCH] Error during video search: {str(e)}")
            return []

    def fetch_transcripts(self, videos: List[Dict]) -> Tuple[List[str], Dict]:
        """Fetch transcripts for the found videos.

        Args:
            videos (List[Dict]): List of video information dictionaries.

        Returns:
            Tuple[List[str], Dict]: (List of successful transcript paths, fetch results)
        """
        if not videos:
            print("[FETCH] No videos to fetch transcripts for")
            return [], {}

        print(f"[FETCH] Starting transcript fetching for {len(videos)} videos")

        # Extract URLs from video data
        urls = [video["url"] for video in videos]

        # Fetch transcripts
        fetch_results = self.transcript_fetcher.fetch_transcripts(urls)

        # Determine successful transcript files
        successful_transcripts = []
        for video in videos:
            video_id = video["video_id"]
            transcript_path = os.path.join(
                self.transcripts_folder, f"{video_id}.{self.transcript_language}.srt"
            )

            if os.path.exists(transcript_path):
                successful_transcripts.append(transcript_path)
                print(f"[FETCH] ✓ Transcript available: {transcript_path}")
            else:
                print(f"[FETCH] ✗ Transcript missing: {transcript_path}")

        print(f"[FETCH] Successfully fetched {len(successful_transcripts)} transcripts")

        # Save fetch metadata
        fetch_metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_videos_processed": len(videos),
            "successful_transcripts": len(successful_transcripts),
            "fetch_results": fetch_results,
            "transcript_files": successful_transcripts,
        }

        metadata_path = os.path.join(
            self.metadata_folder, f"fetch_results_{int(time.time())}.json"
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(fetch_metadata, f, indent=2, ensure_ascii=False)
        print(f"[FETCH] Fetch metadata saved to: {metadata_path}")

        return successful_transcripts, fetch_results

    def summarize_transcripts(
        self, transcript_paths: List[str], videos: List[Dict]
    ) -> Dict:
        """Summarize the fetched transcripts using parallel processing with video context.

        Args:
            transcript_paths (List[str]): List of transcript file paths.
            videos (List[Dict]): List of video information for context.

        Returns:
            Dict: Summarization results.
        """
        if not transcript_paths:
            print("[SUMMARIZE] No transcripts to summarize")
            return {}

        print(
            f"[SUMMARIZE] Starting parallel summarization for {len(transcript_paths)} transcripts"
        )

        # Create a mapping of video IDs to video info for context
        video_info_map = {video["video_id"]: video for video in videos}

        # Use parallel processing with ThreadPoolExecutor for transcript summarization
        from concurrent.futures import ThreadPoolExecutor, as_completed

        summarization_results = {}

        # Determine number of workers (use same logic as individual components)
        num_workers = self.summarizer.num_workers
        if num_workers == 0 or len(transcript_paths) == 1:
            print(
                f"[SUMMARIZE] Using sequential processing for {len(transcript_paths)} transcript(s)"
            )
            # Fall back to sequential processing
            return self._summarize_transcripts_sequential(
                transcript_paths, video_info_map
            )

        print(f"[SUMMARIZE] Using parallel processing with {num_workers} workers")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all summarization tasks
            future_to_path = {}
            for transcript_path in transcript_paths:
                filename = os.path.basename(transcript_path)
                video_id = filename.split(".")[0]
                video_info = video_info_map.get(video_id, {})

                summary_filename = f"{video_id}_summary.json"
                summary_path = os.path.join(self.summaries_folder, summary_filename)

                future = executor.submit(
                    self._summarize_single_transcript,
                    transcript_path,
                    video_info.get("title", ""),
                    video_info.get("description", ""),
                    summary_path,
                    video_id,
                )
                future_to_path[future] = transcript_path

            # Process completed tasks
            for future in as_completed(future_to_path):
                transcript_path = future_to_path[future]
                try:
                    success = future.result()
                    summarization_results[transcript_path] = success
                except Exception as e:
                    print(f"[SUMMARIZE] Error processing {transcript_path}: {str(e)}")
                    summarization_results[transcript_path] = False

        # Log results
        successful_summaries = sum(summarization_results.values())
        print(
            f"[SUMMARIZE] ✓ Completed: {successful_summaries}/{len(transcript_paths)} summaries successful"
        )

        # Save summarization metadata
        summary_metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_transcripts_processed": len(transcript_paths),
            "successful_summaries": sum(summarization_results.values()),
            "summarization_results": summarization_results,
        }

        metadata_path = os.path.join(
            self.metadata_folder, f"summary_results_{int(time.time())}.json"
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(summary_metadata, f, indent=2, ensure_ascii=False)
        print(f"[SUMMARIZE] Summary metadata saved to: {metadata_path}")

        return summarization_results

    def _summarize_single_transcript(
        self,
        transcript_path: str,
        video_title: str,
        video_description: str,
        summary_path: str,
        video_id: str,
    ) -> bool:
        """Summarize a single transcript file.

        Args:
            transcript_path (str): Path to the transcript file.
            video_title (str): Title of the video.
            video_description (str): Description of the video.
            summary_path (str): Path to save the summary.
            video_id (str): Video ID for logging.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            print(f"[SUMMARIZE] Processing: {video_title or video_id}")
            summary = self.summarizer.summarize_transcript(
                transcript_path, video_title, video_description, summary_path
            )
            success = bool(summary)

            if success:
                print(f"[SUMMARIZE] ✓ Summary saved: {os.path.basename(summary_path)}")
            else:
                print(f"[SUMMARIZE] ✗ Failed to summarize: {video_id}")

            return success
        except Exception as e:
            print(f"[SUMMARIZE] Error summarizing {video_id}: {str(e)}")
            return False

    def _summarize_transcripts_sequential(
        self, transcript_paths: List[str], video_info_map: Dict
    ) -> Dict:
        """Summarize transcripts sequentially as fallback.

        Args:
            transcript_paths (List[str]): List of transcript file paths.
            video_info_map (Dict): Mapping of video IDs to video information.

        Returns:
            Dict: Summarization results.
        """
        summarization_results = {}

        for transcript_path in transcript_paths:
            filename = os.path.basename(transcript_path)
            video_id = filename.split(".")[0]
            video_info = video_info_map.get(video_id, {})

            summary_filename = f"{video_id}_summary.json"
            summary_path = os.path.join(self.summaries_folder, summary_filename)

            success = self._summarize_single_transcript(
                transcript_path,
                video_info.get("title", ""),
                video_info.get("description", ""),
                summary_path,
                video_id,
            )
            summarization_results[transcript_path] = success

        return summarization_results

    def run_pipeline(self, search_query: str) -> Dict:
        """Run the complete pipeline with the given search query.

        Args:
            search_query (str): The search query for YouTube videos.

        Returns:
            Dict: Complete pipeline results including all steps.
        """
        pipeline_start_time = time.time()
        print("=" * 80)
        print(f"[PIPELINE] Starting complete YouTube processing pipeline")
        print(f"[PIPELINE] Search Query: '{search_query}'")
        print(f"[PIPELINE] Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Step 1: Search for videos
        print("\n" + "=" * 50)
        print("STEP 1: VIDEO SEARCH")
        print("=" * 50)
        videos = self.search_videos(search_query)

        if not videos:
            print("[PIPELINE] ❌ No videos found. Pipeline terminated.")
            return {
                "success": False,
                "error": "No videos found for the search query",
                "search_query": search_query,
                "videos_found": 0,
            }

        # Step 2: Fetch transcripts
        print("\n" + "=" * 50)
        print("STEP 2: TRANSCRIPT FETCHING")
        print("=" * 50)
        transcript_paths, fetch_results = self.fetch_transcripts(videos)

        if not transcript_paths:
            print("[PIPELINE] ❌ No transcripts could be fetched. Pipeline terminated.")
            return {
                "success": False,
                "error": "No transcripts could be fetched",
                "search_query": search_query,
                "videos_found": len(videos),
                "transcripts_fetched": 0,
            }

        # Step 3: Summarize transcripts
        print("\n" + "=" * 50)
        print("STEP 3: TRANSCRIPT SUMMARIZATION")
        print("=" * 50)
        summarization_results = self.summarize_transcripts(transcript_paths, videos)

        # Calculate final results
        pipeline_end_time = time.time()
        pipeline_duration = pipeline_end_time - pipeline_start_time
        successful_summaries = sum(summarization_results.values())

        # Create final results
        final_results = {
            "success": True,
            "search_query": search_query,
            "pipeline_duration_seconds": round(pipeline_duration, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "videos_found": len(videos),
            "transcripts_fetched": len(transcript_paths),
            "summaries_created": successful_summaries,
            "output_folder": self.output_folder,
            "transcripts_folder": self.transcripts_folder,
            "summaries_folder": self.summaries_folder,
            "videos": videos,
            "transcript_paths": transcript_paths,
            "summarization_results": summarization_results,
        }

        # Save final results
        results_path = os.path.join(
            self.metadata_folder, f"pipeline_results_{int(time.time())}.json"
        )
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        # Print final summary
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED")
        print("=" * 80)
        print(f"[PIPELINE] ✓ Search Query: '{search_query}'")
        print(f"[PIPELINE] ✓ Videos Found: {len(videos)}")
        print(f"[PIPELINE] ✓ Transcripts Fetched: {len(transcript_paths)}")
        print(f"[PIPELINE] ✓ Summaries Created: {successful_summaries}")
        print(f"[PIPELINE] ✓ Total Duration: {pipeline_duration:.2f} seconds")
        print(f"[PIPELINE] ✓ Results Saved: {results_path}")
        print(f"[PIPELINE] ✓ Output Folder: {self.output_folder}")
        print("=" * 80)

        return final_results


def parse_arguments():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Complete YouTube video processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python youtube_pipeline.py --query "Python programming tutorial"
  python youtube_pipeline.py --query "machine learning basics" --max-videos 3
  python youtube_pipeline.py --query "React hooks" --output-folder "react_summaries"
  python youtube_pipeline.py --query "Docker tutorial" --workers 4 --language "en"
        """,
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Search query for YouTube videos (e.g., 'Python Programming Tutorial')",
    )

    parser.add_argument(
        "--max-videos",
        "-m",
        type=int,
        help="Maximum number of videos to process (overrides config default)",
    )

    parser.add_argument(
        "--output-folder",
        "-o",
        type=str,
        help="Base output folder for all results (default: 'pipeline_output')",
    )

    parser.add_argument(
        "--language",
        "-l",
        type=str,
        help="Language code for transcripts (e.g., 'en', 'es', 'fr')",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        help="Number of concurrent workers for parallel processing",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """Main function for the YouTube processing pipeline."""
    args = parse_arguments()

    print("🎬 YouTube Video Processing Pipeline")
    print("=" * 80)

    # Validate environment variables
    required_env_vars = ["YOUTUBE_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(
            f"❌ Error: Missing required environment variables: {', '.join(missing_vars)}"
        )
        print("\nSetup Instructions:")
        print("1. Get YouTube API key: https://console.cloud.google.com/")
        print("2. Get OpenAI API key: https://platform.openai.com/api-keys")
        print("3. Set environment variables:")
        for var in missing_vars:
            print(f"   export {var}='your_key_here'")
        print("4. Or create a .env file with these variables")
        sys.exit(1)

    try:
        # Initialize pipeline
        print(f"🔧 Initializing pipeline...")
        pipeline = YouTubePipeline(
            max_videos=args.max_videos,
            transcript_language=args.language,
            output_folder=args.output_folder,
            num_workers=args.workers,
        )

        # Run the complete pipeline
        print(f"🚀 Starting pipeline for query: '{args.query}'")
        results = pipeline.run_pipeline(args.query)

        if results["success"]:
            print(f"\n✅ Pipeline completed successfully!")
            print(f"📁 Check results in: {results['output_folder']}")
        else:
            print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {str(e)}")
        if args.verbose:
            import traceback

            print("\nDetailed error:")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
