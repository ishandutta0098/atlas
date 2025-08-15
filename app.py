"""
YouTube Processing Pipeline - Gradio Web Interface

This script provides a web-based interface for the complete YouTube processing pipeline using Gradio.
It integrates all components (video search, transcript fetching, summarization) into a user-friendly web app.

Key Features:
- YouTube video search with natural language queries
- Transcript fetching from found videos
- AI-powered summarization for technical content
- Step-by-step pipeline visualization
- Real-time processing status updates
- Support for multiple concurrent workers
- Professional web interface with custom styling

Required Dependencies:
- gradio: Web interface framework
- All dependencies from the YouTube pipeline components
- OpenAI API key for summarization
- YouTube Data API key for video search

Commands to run:
# Launch Gradio web interface
python app_youtube.py

# Launch with custom configuration
python app_youtube.py --host 0.0.0.0 --port 8080 --share

# Launch with debug mode
python app_youtube.py --debug
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

# Import components from the YouTube pipeline
from src.youtube_pipeline import YouTubePipeline

load_dotenv()


def is_cloud_environment():
    """
    Checks if the code is running in a cloud environment (Hugging Face Spaces, Colab, etc.).

    Returns:
        bool: True if running in cloud environment, False otherwise.
    """
    # Check for various cloud environment indicators
    cloud_indicators = [
        os.environ.get("SYSTEM") == "spaces",  # Hugging Face Spaces
        os.environ.get("COLAB_GPU") is not None,  # Google Colab
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None,  # Kaggle
        os.path.exists("/.dockerenv"),  # Docker container
    ]
    return any(cloud_indicators)


def validate_api_keys():
    """
    Validate that required API keys are available.

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    required_keys = {
        "OPENAI_API_KEY": "OpenAI API key for transcript summarization",
        "YOUTUBE_API_KEY": "YouTube Data API key for video search",
    }

    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"- {key}: {description}")

    if missing_keys:
        error_msg = "❌ Missing required API keys:\n" + "\n".join(missing_keys)
        error_msg += (
            "\n\nPlease set these environment variables or add them to a .env file."
        )
        return False, error_msg

    return True, ""


def gradio_youtube_pipeline(
    search_query: str,
    max_videos: int,
    transcript_language: str,
    num_workers: int,
    openai_api_key: str,
    youtube_api_key: str,
    use_env_keys: bool,
) -> Tuple[str, str, str, str, str, List]:
    """
    Gradio interface function for the YouTube processing pipeline.

    Args:
        search_query (str): The search query for YouTube videos
        max_videos (int): Maximum number of videos to process
        transcript_language (str): Language code for transcripts
        num_workers (int): Number of concurrent workers
        openai_api_key (str): OpenAI API key (if not using env)
        youtube_api_key (str): YouTube API key (if not using env)
        use_env_keys (bool): Whether to use environment variables for API keys

    Returns:
        Tuple containing: (final_summary, search_results, transcript_status,
                          summarization_status, pipeline_metadata, file_gallery)
    """
    try:
        # Validate inputs
        if not search_query or not search_query.strip():
            return ("❌ Error: Please provide a search query.", "", "", "", "", [])

        # Handle API keys
        if use_env_keys:
            # Validate environment keys
            is_valid, error_msg = validate_api_keys()
            if not is_valid:
                return (error_msg, "", "", "", "", [])
        else:
            # Use provided API keys
            if not openai_api_key or not openai_api_key.strip():
                return ("❌ Error: OpenAI API key is required.", "", "", "", "", [])
            if not youtube_api_key or not youtube_api_key.strip():
                return ("❌ Error: YouTube API key is required.", "", "", "", "", [])

            # Set the API keys as environment variables for the pipeline
            os.environ["OPENAI_API_KEY"] = openai_api_key
            os.environ["YOUTUBE_API_KEY"] = youtube_api_key

        # Initialize the pipeline with user configuration
        pipeline = YouTubePipeline(
            max_videos=max_videos,
            transcript_language=transcript_language,
            output_folder=f"pipeline_output_{int(time.time())}",
            num_workers=num_workers,
        )

        # Run the complete pipeline
        start_time = time.time()
        results = pipeline.run_pipeline(search_query)
        end_time = time.time()

        if not results.get("success", False):
            error_msg = results.get("error", "Unknown pipeline error")
            return (f"❌ Pipeline failed: {error_msg}", "", "", "", "", [])

        # Format the final summary
        final_summary = format_final_summary(results, end_time - start_time)

        # Format search results
        search_results = format_search_results(results.get("videos", []))

        # Format transcript status
        transcript_status = format_transcript_status(
            results.get("transcript_paths", []), results.get("videos", [])
        )

        # Format summarization status
        summarization_status = format_summarization_status(
            results.get("summarization_results", {})
        )

        # Format pipeline metadata
        pipeline_metadata = format_pipeline_metadata(results)

        # Create file gallery (summary files)
        file_gallery = create_file_gallery(results)

        return (
            final_summary,
            search_results,
            transcript_status,
            summarization_status,
            pipeline_metadata,
            file_gallery,
        )

    except Exception as e:
        return (f"❌ Pipeline Error: {str(e)}", "", "", "", "", [])


def format_final_summary(results: Dict, duration: float) -> str:
    """Format the final pipeline summary."""
    summary = f"""
🎬 **YouTube Processing Pipeline Completed Successfully!**

**Search Query:** {results.get('search_query', 'N/A')}
**Total Duration:** {duration:.2f} seconds

**📊 Results Summary:**
• Videos Found: {results.get('videos_found', 0)}
• Transcripts Fetched: {results.get('transcripts_fetched', 0)}
• Summaries Created: {results.get('summaries_created', 0)}

**📁 Output Locations:**
• Main Output: `{results.get('output_folder', 'N/A')}`
• Transcripts: `{results.get('transcripts_folder', 'N/A')}`
• Summaries: `{results.get('summaries_folder', 'N/A')}`

✅ All pipeline steps completed successfully. Check the tabs below for detailed information about each step.
"""
    return summary.strip()


def format_search_results(videos: List[Dict]) -> str:
    """Format the video search results."""
    if not videos:
        return "❌ No videos found for the search query."

    results = f"🔍 **Found {len(videos)} videos:**\n\n"

    for i, video in enumerate(videos, 1):
        title = video.get("title", "Unknown Title")
        channel = video.get("channel_title", "Unknown Channel")
        url = video.get("url", "#")
        description = video.get("description", "")
        publish_date = video.get("published_at", "Unknown Date")

        # Truncate description if too long
        if len(description) > 200:
            description = description[:200] + "..."

        results += f"""
**{i}. {title}**
   • **Channel:** {channel}
   • **Published:** {publish_date}
   • **URL:** {url}
   • **Description:** {description}

"""

    return results


def format_transcript_status(transcript_paths: List[str], videos: List[Dict]) -> str:
    """Format the transcript fetching status."""
    if not videos:
        return "❌ No videos to fetch transcripts from."

    status = f"📝 **Transcript Fetching Results:**\n\n"
    status += f"• Total videos processed: {len(videos)}\n"
    status += f"• Successful transcripts: {len(transcript_paths)}\n\n"

    # Create a mapping of video IDs to transcripts
    transcript_files = {Path(tp).stem.split(".")[0]: tp for tp in transcript_paths}

    for i, video in enumerate(videos, 1):
        title = video.get("title", "Unknown Title")
        video_id = video.get("video_id", "")

        if video_id in transcript_files:
            status += f"✅ **{i}. {title}**\n   Transcript: `{Path(transcript_files[video_id]).name}`\n\n"
        else:
            status += f"❌ **{i}. {title}**\n   Transcript: Failed to fetch\n\n"

    return status


def format_summarization_status(summarization_results: Dict) -> str:
    """Format the summarization results."""
    if not summarization_results:
        return "❌ No summarization results available."

    total_files = len(summarization_results)
    successful = sum(summarization_results.values())
    failed = total_files - successful

    status = f"🤖 **AI Summarization Results:**\n\n"
    status += f"• Total transcripts processed: {total_files}\n"
    status += f"• Successful summaries: {successful}\n"
    status += f"• Failed summaries: {failed}\n\n"

    for transcript_path, success in summarization_results.items():
        filename = Path(transcript_path).name
        if success:
            status += f"✅ **{filename}** → Summary created\n"
        else:
            status += f"❌ **{filename}** → Summarization failed\n"

    return status


def format_pipeline_metadata(results: Dict) -> str:
    """Format detailed pipeline metadata."""
    metadata = f"""
🔧 **Pipeline Configuration & Metadata:**

**Timing Information:**
• Start Time: {results.get('timestamp', 'N/A')}
• Total Duration: {results.get('pipeline_duration_seconds', 0):.2f} seconds

**Processing Statistics:**
• Videos Found: {results.get('videos_found', 0)}
• Transcripts Fetched: {results.get('transcripts_fetched', 0)}
• Summaries Created: {results.get('summaries_created', 0)}
• Success Rate: {(results.get('summaries_created', 0) / max(results.get('videos_found', 1), 1) * 100):.1f}%

**Output Structure:**
• Main Output Folder: `{results.get('output_folder', 'N/A')}`
• Transcripts Folder: `{results.get('transcripts_folder', 'N/A')}`
• Summaries Folder: `{results.get('summaries_folder', 'N/A')}`

**File Paths:**
"""

    # Add transcript file paths
    transcript_paths = results.get("transcript_paths", [])
    if transcript_paths:
        metadata += "\n**Transcript Files:**\n"
        for path in transcript_paths:
            metadata += f"• `{path}`\n"

    return metadata


def create_file_gallery(results: Dict) -> List[Tuple[str, str]]:
    """Create a gallery of output files (currently returns empty as we're dealing with text files)."""
    # For now, return empty gallery since we're dealing with text files
    # In the future, this could show preview images of the summaries or other visual representations
    return []


def create_gradio_app():
    """Create and configure the Gradio application."""

    # Custom CSS for professional styling
    css = """
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 20px !important;
        background: #f8f9fa;
    }
    
    /* Header styling */
    .header-section {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .header-section h1 {
        font-size: 2.5em;
        margin-bottom: 15px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-section p {
        font-size: 1.2em;
        margin-bottom: 10px;
        opacity: 0.95;
    }
    
    /* Input section styling */
    .input-section {
        background: #ffffff;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    
    /* Results section styling */
    .results-section {
        background: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    
    /* Section headers */
    .section-header {
        color: #667eea;
        font-weight: bold;
        border-bottom: 2px solid #667eea;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }
    
    /* Button styling */
    .primary-button {
        width: 100%;
        padding: 12px;
        font-size: 16px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .primary-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* API key styling */
    .api-key-input {
        border: 2px solid #fbbf24 !important;
        background: #fffbeb !important;
    }
    
    /* Step boxes */
    .step-box {
        background: #ffffff;
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .search-step { border-left: 4px solid #10b981; }
    .transcript-step { border-left: 4px solid #3b82f6; }
    .summary-step { border-left: 4px solid #8b5cf6; }
    """

    with gr.Blocks(css=css, title="🎬 YouTube Processing Pipeline") as app:
        # Header Section
        with gr.Row():
            with gr.Column(elem_classes=["header-section"]):
                gr.HTML(
                    """
                <div style="text-align: center;">
                    <h1>🎬 YouTube Processing Pipeline</h1>
                    <p>Comprehensive AI-powered pipeline for YouTube video analysis and summarization</p>
                    <div style="display: flex; justify-content: center; gap: 25px; margin-top: 20px; flex-wrap: wrap;">
                        <div>🔍 <strong>Video Search</strong><br/>YouTube Data API integration</div>
                        <div>📝 <strong>Transcript Fetching</strong><br/>Automatic subtitle extraction</div>
                        <div>🤖 <strong>AI Summarization</strong><br/>Technical content analysis</div>
                        <div>⚡ <strong>Parallel Processing</strong><br/>Concurrent workflow execution</div>
                        <div>📊 <strong>Detailed Tracking</strong><br/>Step-by-step visualization</div>
                    </div>
                </div>
                """
                )

        with gr.Row(equal_height=False):
            # Left Column - Input Configuration
            with gr.Column(scale=1, elem_classes=["input-section"]):
                gr.HTML('<h3 class="section-header">🔍 Search Configuration</h3>')

                search_query = gr.Textbox(
                    label="YouTube Search Query",
                    placeholder="e.g., 'Python machine learning tutorial', 'React best practices', 'Docker deployment guide'",
                    lines=3,
                    info="Enter your search query to find relevant YouTube videos",
                )

                with gr.Row():
                    max_videos = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Max Videos",
                        info="Maximum number of videos to process",
                    )

                    num_workers = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=2,
                        step=1,
                        label="Concurrent Workers",
                        info="Number of parallel workers for processing",
                    )

                transcript_language = gr.Dropdown(
                    choices=[
                        "en",
                        "es",
                        "fr",
                        "de",
                        "it",
                        "pt",
                        "ru",
                        "ja",
                        "ko",
                        "zh",
                    ],
                    value="en",
                    label="Transcript Language",
                    info="Language code for subtitle extraction",
                )

                gr.HTML('<h3 class="section-header">🔑 API Configuration</h3>')

                use_env_keys = gr.Checkbox(
                    value=True,
                    label="Use Environment Variables for API Keys",
                    info="Check if you have OPENAI_API_KEY and YOUTUBE_API_KEY set as environment variables",
                )

                with gr.Column(visible=False) as api_key_inputs:
                    openai_api_key = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder="sk-...",
                        info="Required for transcript summarization",
                    )

                    youtube_api_key = gr.Textbox(
                        label="YouTube Data API Key",
                        type="password",
                        placeholder="AIza...",
                        info="Required for video search",
                    )

                # Show/hide API key inputs based on checkbox
                def toggle_api_inputs(use_env):
                    return gr.update(visible=not use_env)

                use_env_keys.change(
                    fn=toggle_api_inputs,
                    inputs=[use_env_keys],
                    outputs=[api_key_inputs],
                )

                process_btn = gr.Button(
                    "🚀 Start Pipeline",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-button"],
                )

            # Right Column - Results Summary
            with gr.Column(scale=2, elem_classes=["results-section"]):
                gr.HTML('<h3 class="section-header">📋 Pipeline Results</h3>')

                final_summary = gr.Textbox(
                    label="Final Summary",
                    lines=12,
                    max_lines=20,
                    show_copy_button=True,
                    info="Overall pipeline results and output locations",
                )

        # Pipeline Steps - Detailed breakdown in tabs
        gr.HTML(
            '<h2 style="text-align: center; color: #667eea; margin: 30px 0 20px 0;">🔍 Pipeline Step Details</h2>'
        )

        with gr.Tabs():
            with gr.TabItem("🔍 Step 1: Video Search", elem_classes=["search-step"]):
                search_results = gr.Textbox(
                    label="YouTube Video Search Results",
                    lines=15,
                    max_lines=25,
                    show_copy_button=True,
                    info="Found videos with titles, channels, descriptions, and URLs",
                )

            with gr.TabItem(
                "📝 Step 2: Transcript Fetching", elem_classes=["transcript-step"]
            ):
                transcript_status = gr.Textbox(
                    label="Transcript Extraction Status",
                    lines=15,
                    max_lines=25,
                    show_copy_button=True,
                    info="Status of transcript fetching for each video",
                )

            with gr.TabItem(
                "🤖 Step 3: AI Summarization", elem_classes=["summary-step"]
            ):
                summarization_status = gr.Textbox(
                    label="AI Summarization Results",
                    lines=15,
                    max_lines=25,
                    show_copy_button=True,
                    info="Results of AI-powered transcript summarization",
                )

            with gr.TabItem("🔧 Pipeline Metadata"):
                pipeline_metadata = gr.Textbox(
                    label="Detailed Pipeline Information",
                    lines=15,
                    max_lines=25,
                    show_copy_button=True,
                    info="Configuration, timing, and file path information",
                )

        # File Gallery (for future use)
        file_gallery = gr.Gallery(
            label="Output Files",
            show_label=False,
            visible=False,  # Hidden for now since we're dealing with text files
            columns=3,
            rows=1,
        )

        # Event handler
        process_btn.click(
            fn=gradio_youtube_pipeline,
            inputs=[
                search_query,
                max_videos,
                transcript_language,
                num_workers,
                openai_api_key,
                youtube_api_key,
                use_env_keys,
            ],
            outputs=[
                final_summary,
                search_results,
                transcript_status,
                summarization_status,
                pipeline_metadata,
                file_gallery,
            ],
        )

    return app


if __name__ == "__main__":
    # Check for cloud environment
    is_cloud = is_cloud_environment()

    if is_cloud:
        print("☁️  Detected cloud environment")
        print("📋 Using default configuration for cloud deployment...")

        # Use default values for cloud environments
        class DefaultArgs:
            def __init__(self):
                self.host = "0.0.0.0"
                self.port = 7860
                self.share = False
                self.debug = False

        args = DefaultArgs()

    else:
        print("💻 Detected local environment")
        print("📋 Using command line arguments...")

        # Parse command line arguments for local development
        parser = argparse.ArgumentParser(
            description="Gradio Web Interface for YouTube Processing Pipeline"
        )
        parser.add_argument(
            "--host", type=str, default="127.0.0.1", help="Host to run the server on"
        )
        parser.add_argument(
            "--port", type=int, default=7860, help="Port to run the server on"
        )
        parser.add_argument("--share", action="store_true", help="Create a public link")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")

        args = parser.parse_args()

    print("🔧 Configuration:")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Share: {args.share}")
    print(f"   Debug: {args.debug}")

    print("🚀 Launching Gradio interface...")

    app = create_gradio_app()
    app.launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
        show_error=args.debug,
    )

    if not is_cloud:
        print("\nExample usage:")
        print("  # Launch with default settings")
        print("  python app_youtube.py")
        print("  # Launch with public sharing")
        print("  python app_youtube.py --share")
        print("  # Launch on custom port")
        print("  python app_youtube.py --port 8080")
    else:
        print("\n☁️  Running in cloud environment - configuration is automatic!")
