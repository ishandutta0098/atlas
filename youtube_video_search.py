#!/usr/bin/env python3
"""
YouTube Video Search Script using YouTube Data API

This script searches for YouTube videos based on a query using the YouTube Data API.
It finds relevant YouTube videos matching your search query.
"""

import argparse
import os

from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()


def search_youtube_videos_api(search_query, max_results=10):
    """
    Search for YouTube videos using the YouTube Data API.

    Args:
        search_query (str): The search query to find relevant YouTube videos
        max_results (int): Maximum number of results to return

    Returns:
        list: List of video information dictionaries
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError(
            "YouTube API key is required. Set YOUTUBE_API_KEY environment variable."
        )

    try:
        # Build the YouTube API client
        youtube = build("youtube", "v3", developerKey=api_key)

        # Call the search.list method to retrieve results matching the query
        search_response = (
            youtube.search()
            .list(
                q=search_query,
                part="id,snippet",
                maxResults=max_results,
                type="video",
                order="relevance",
            )
            .execute()
        )

        videos = []
        for search_result in search_response.get("items", []):
            # Handle different types of search results (videos vs playlists/channels)
            if "id" in search_result:
                if (
                    isinstance(search_result["id"], dict)
                    and "videoId" in search_result["id"]
                ):
                    video_id = search_result["id"]["videoId"]
                elif isinstance(search_result["id"], str):
                    video_id = search_result["id"]
                else:
                    # Skip non-video results (playlists, channels, etc.)
                    continue
            else:
                continue

            video_info = {
                "title": search_result["snippet"]["title"],
                "channel": search_result["snippet"]["channelTitle"],
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "description": (
                    search_result["snippet"]["description"][:200] + "..."
                    if len(search_result["snippet"]["description"]) > 200
                    else search_result["snippet"]["description"]
                ),
                "published_at": search_result["snippet"]["publishedAt"],
                "video_id": video_id,
            }
            videos.append(video_info)

        return videos

    except Exception as e:
        raise Exception(f"Error searching YouTube videos: {str(e)}")


def format_video_results(videos, search_query):
    """
    Format video search results into a readable string.

    Args:
        videos (list): List of video dictionaries
        search_query (str): The original search query

    Returns:
        str: Formatted search results
    """
    if not videos:
        return f"No YouTube videos found for query: '{search_query}'"

    result = f"Found {len(videos)} YouTube videos for '{search_query}':\n\n"

    for i, video in enumerate(videos, 1):
        result += f"{i}. **{video['title']}**\n"
        result += f"   Channel: {video['channel']}\n"
        result += f"   URL: {video['url']}\n"
        result += f"   Description: {video['description']}\n"
        result += f"   Published: {video['published_at']}\n\n"

    return result


def search_youtube_videos(search_query, max_results=10):
    """
    Searches for YouTube videos based on a query using the YouTube Data API.

    Args:
        search_query (str): The search query to find relevant YouTube videos
        max_results (int): Maximum number of results to return

    Returns:
        str: Formatted search results
    """
    try:
        # Get videos using the YouTube API
        videos = search_youtube_videos_api(search_query, max_results)

        # Format and return the results
        return format_video_results(videos, search_query)

    except Exception as e:
        return f"Error searching YouTube videos: {str(e)}"


def search_youtube_direct(search_query, max_results=10):
    """
    Direct YouTube search - returns raw video data.

    Args:
        search_query (str): The search query to find relevant YouTube videos
        max_results (int): Maximum number of results to return

    Returns:
        list: List of video information dictionaries
    """
    return search_youtube_videos_api(search_query, max_results)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Search for YouTube videos using CrewAI's web search tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python youtube_video_search.py --query "Python Programming Tutorial"
  python youtube_video_search.py --query "machine learning basics"
  python youtube_video_search.py --query "React.js hooks tutorial" --verbose
        """,
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Search query to find YouTube videos (e.g., 'Python Programming Tutorial')",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """
    Main function demonstrating the YouTube video search functionality.
    """
    args = parse_arguments()
    search_query = args.query

    print("🎥 YouTube Video Search")
    print("=" * 50)
    print(f"Search Query: {search_query}")
    if args.verbose:
        print(f"Verbose mode: enabled")
    print("=" * 50)

    try:
        # Perform the search
        print("🔍 Searching for YouTube videos...")
        result = search_youtube_videos(search_query)

        print("\n📋 Search Results:")
        print("-" * 30)
        print(result)

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up your YouTube API key: export YOUTUBE_API_KEY='your_key'")
        print("3. Check your internet connection")
        print("4. Ensure your YouTube API quota is not exceeded")
        print("5. Get your YouTube API key at: https://console.cloud.google.com/")

        if args.verbose:
            import traceback

            print("\nDetailed error:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
