import os
from typing import List, Optional

from yt_dlp import YoutubeDL


class YouTubeTranscriptFetcher:
    """A class to fetch transcripts from YouTube videos.

    This class provides functionality to download transcripts (both human-generated
    and automatic) from YouTube videos using yt-dlp library.
    """

    def __init__(self, output_folder: str = "transcripts", language: str = "en"):
        """Initialize the YouTubeTranscriptFetcher.

        Args:
            output_folder (str): Directory where transcripts will be saved.
                Defaults to "transcripts".
            language (str): Language code for subtitles. Defaults to "en".
        """
        self.output_folder = output_folder
        self.language = language
        self._ensure_output_folder()

    def _ensure_output_folder(self) -> None:
        """Create the output folder if it doesn't exist."""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def _get_ydl_opts(self) -> dict:
        """Get the yt-dlp options configuration.

        Returns:
            dict: Configuration options for yt-dlp.
        """
        return {
            "skip_download": True,
            "writesubtitles": True,  # human captions
            "writeautomaticsub": True,  # auto captions
            "subtitleslangs": [self.language],
            "subtitlesformat": "srt",
            "outtmpl": os.path.join(self.output_folder, "%(id)s.%(ext)s"),
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

    def fetch_transcripts(self, urls: List[str]) -> dict:
        """Fetch transcripts for multiple YouTube videos.

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


# Example usage
if __name__ == "__main__":
    # Example URLs
    urls = [
        "https://www.youtube.com/watch?v=UV81LAb3x2g",
        "https://www.youtube.com/watch?v=q6kJ71tEYqM",
    ]

    # Initialize the fetcher
    fetcher = YouTubeTranscriptFetcher(output_folder="transcripts")

    # Fetch transcripts
    results = fetcher.fetch_transcripts(urls)

    # Print results
    for url, success in results.items():
        status = "Success" if success else "Failed"
        print(f"{url}: {status}")
