import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from yt_dlp import YoutubeDL

from utils import ensure_output_folder, get_config, get_worker_count, setup_logging


class YouTubeTranscriptFetcher:
    """A class to fetch transcripts from YouTube videos.

    This class provides functionality to download transcripts (both human-generated
    and automatic) from YouTube videos using yt-dlp library.
    """

    def __init__(
        self,
        output_folder: Optional[str] = None,
        language: Optional[str] = None,
        num_workers: Optional[int] = None,
    ):
        """Initialize the YouTubeTranscriptFetcher.

        Args:
            output_folder (Optional[str]): Directory where transcripts will be saved.
                If None, uses config default.
            language (Optional[str]): Language code for subtitles. If None, uses config default.
            num_workers (Optional[int]): Number of concurrent workers for parallel processing.
                If None, auto-detects based on config and CPU count.
        """
        # Initialize configuration and logging
        setup_logging()

        # Set configuration values
        self.output_folder = output_folder or get_config(
            "processing.transcripts.output_folder", "transcripts"
        )
        self.language = language or get_config("processing.transcripts.language", "en")
        self.num_workers = get_worker_count(num_workers)

        # Ensure output folder exists
        self.output_folder = ensure_output_folder(self.output_folder)

    def _get_ydl_opts(self) -> dict:
        """Get the yt-dlp options configuration.

        Returns:
            dict: Configuration options for yt-dlp.
        """
        return {
            "skip_download": get_config("download.skip_download", True),
            "writesubtitles": get_config(
                "download.write_subtitles", True
            ),  # human captions
            "writeautomaticsub": get_config(
                "download.write_automatic_sub", True
            ),  # auto captions
            "subtitleslangs": [self.language],
            "subtitlesformat": get_config("download.subtitles_format", "srt"),
            "outtmpl": os.path.join(
                self.output_folder,
                get_config("download.output_template", "%(id)s.%(ext)s"),
            ),
        }

    def fetch_transcript(self, url: str) -> bool:
        """Fetch transcript for a single YouTube video.

        Args:
            url (str): YouTube video URL.

        Returns:
            bool: True if transcript was successfully downloaded, False otherwise.
        """
        try:
            with YoutubeDL(self._get_ydl_opts()) as ydl:
                ydl.download([url])
            return True
        except Exception as e:
            print(f"Error downloading transcript for {url}: {str(e)}")
            return False

    def _fetch_transcripts_sequential(self, urls: List[str]) -> dict:
        """Fetch transcripts sequentially (one at a time).

        Args:
            urls (List[str]): List of YouTube video URLs.

        Returns:
            dict: Dictionary with URLs as keys and success status as values.
        """
        results = {}
        for url in urls:
            print(f"Fetching transcript for: {url}")
            results[url] = self.fetch_transcript(url)
        return results

    def _fetch_transcripts_parallel(self, urls: List[str]) -> dict:
        """Fetch transcripts in parallel using ThreadPoolExecutor.

        Args:
            urls (List[str]): List of YouTube video URLs.

        Returns:
            dict: Dictionary with URLs as keys and success status as values.
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all download tasks
            future_to_url = {
                executor.submit(self.fetch_transcript, url): url for url in urls
            }

            # Process completed tasks
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    success = future.result()
                    results[url] = success
                    status = "Success" if success else "Failed"
                    print(f"Completed {url}: {status}")
                except Exception as e:
                    results[url] = False
                    print(f"Error processing {url}: {str(e)}")

        return results

    def fetch_transcripts(self, urls: List[str]) -> dict:
        """Fetch transcripts for multiple YouTube videos with automatic parallel/sequential fallback.

        Args:
            urls (List[str]): List of YouTube video URLs.

        Returns:
            dict: Dictionary with URLs as keys and success status as values.
        """
        if not urls:
            return {}

        # Use sequential processing if num_workers is 0 or only one URL
        if self.num_workers == 0 or len(urls) == 1:
            print(f"Using sequential processing for {len(urls)} URL(s)")
            return self._fetch_transcripts_sequential(urls)

        # Try parallel processing first
        try:
            print(
                f"Using parallel processing with {self.num_workers} workers for {len(urls)} URLs"
            )
            return self._fetch_transcripts_parallel(urls)
        except Exception as e:
            print(
                f"Parallel processing failed ({str(e)}), falling back to sequential processing"
            )
            return self._fetch_transcripts_sequential(urls)


# Example usage
if __name__ == "__main__":
    # Example URLs
    urls = [
        "https://www.youtube.com/watch?v=UV81LAb3x2g",
        # "https://www.youtube.com/watch?v=q6kJ71tEYqM",
        # "https://www.youtube.com/watch?v=gpz6C_2l5jI",
    ]

    # Initialize the fetcher
    fetcher = YouTubeTranscriptFetcher(output_folder="transcripts")

    # Fetch transcripts
    results = fetcher.fetch_transcripts(urls)

    # Print results
    for url, success in results.items():
        status = "Success" if success else "Failed"
        print(f"{url}: {status}")
