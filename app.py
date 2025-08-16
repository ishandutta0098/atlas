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


# Global variable to store pipeline state between functions
pipeline_state = {
    "pipeline": None,
    "videos": None,
    "transcript_paths": None,
    "search_results": "",
    "transcripts_output": "",
    "summaries_output": "",
    "comparison_table": "",
    "assignments_output": "",
}


def step1_search_videos(
    search_query: str,
    max_videos: int,
    transcript_language: str,
    num_workers: int,
    openai_api_key: str,
    youtube_api_key: str,
    use_env_keys: bool,
    progress: gr.Progress = gr.Progress(),
) -> str:
    """
    Step 1: Search for YouTube videos.

    Returns:
        str: Formatted search results
    """
    try:
        # Reset pipeline state
        global pipeline_state
        pipeline_state = {
            "pipeline": None,
            "videos": None,
            "transcript_paths": None,
            "search_results": "",
            "transcripts_output": "",
            "summaries_output": "",
            "comparison_table": "",
            "assignments_output": "",
        }

        # Validate inputs
        if not search_query or not search_query.strip():
            return "❌ Error: Please provide a search query."

        # Handle API keys
        if use_env_keys:
            # Validate environment keys
            is_valid, error_msg = validate_api_keys()
            if not is_valid:
                return error_msg
        else:
            # Use provided API keys
            if not openai_api_key or not openai_api_key.strip():
                return "❌ Error: OpenAI API key is required."
            if not youtube_api_key or not youtube_api_key.strip():
                return "❌ Error: YouTube API key is required."

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

        # Store pipeline in global state
        pipeline_state["pipeline"] = pipeline

        # Search for videos with progress tracking
        progress(0.1, desc="🔍 Searching for YouTube videos...")
        videos = pipeline.search_videos(search_query)

        if not videos:
            return "❌ No videos found for the search query."

        # Store videos in global state
        pipeline_state["videos"] = videos

        # Format and store search results
        search_results = format_search_results(videos)
        pipeline_state["search_results"] = search_results

        progress(1.0, desc="✅ Video search completed!")
        return search_results

    except Exception as e:
        print(f"[ERROR] Video search failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return f"❌ Video Search Error: {str(e)}"


def step2_fetch_transcripts(
    search_results_input: str, progress: gr.Progress = gr.Progress()
) -> str:
    """
    Step 2: Fetch transcripts for found videos.

    Args:
        search_results_input: Previous step output (not used, just for chaining)

    Returns:
        str: Formatted transcript results
    """
    try:
        global pipeline_state

        # Check if we have valid state from previous step
        if not pipeline_state["pipeline"] or not pipeline_state["videos"]:
            return "❌ Error: No valid pipeline state. Please run video search first."

        pipeline = pipeline_state["pipeline"]
        videos = pipeline_state["videos"]

        # Fetch transcripts with progress tracking
        progress(0.1, desc="📝 Fetching video transcripts...")
        transcript_paths, fetch_results = pipeline.fetch_transcripts(videos)

        # Store transcript paths in global state
        pipeline_state["transcript_paths"] = transcript_paths

        # Format transcript results
        transcripts_output = format_transcript_results(
            transcript_paths, videos, pipeline.transcripts_folder
        )
        pipeline_state["transcripts_output"] = transcripts_output

        progress(1.0, desc="✅ Transcript fetching completed!")
        return transcripts_output

    except Exception as e:
        print(f"[ERROR] Transcript fetching failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return f"❌ Transcript Fetching Error: {str(e)}"


def step3_generate_summaries(
    transcripts_input: str, progress: gr.Progress = gr.Progress()
) -> str:
    """
    Step 3: Generate AI summaries for transcripts.

    Args:
        transcripts_input: Previous step output (not used, just for chaining)

    Returns:
        str: Formatted summaries results
    """
    try:
        global pipeline_state

        # Check if we have valid state from previous steps
        if (
            not pipeline_state["pipeline"]
            or not pipeline_state["videos"]
            or not pipeline_state["transcript_paths"]
        ):
            return "❌ Error: No valid pipeline state. Please run previous steps first."

        pipeline = pipeline_state["pipeline"]
        videos = pipeline_state["videos"]
        transcript_paths = pipeline_state["transcript_paths"]

        if transcript_paths:
            # Generate summaries with progress tracking
            progress(0.1, desc="🤖 Generating AI summaries...")

            # Use the pipeline's built-in parallel summarization
            summarization_results = pipeline.summarize_transcripts(
                transcript_paths, videos
            )

            # Format summaries results
            summaries_output = format_summaries_results(
                transcript_paths, videos, pipeline.summaries_folder
            )
            pipeline_state["summaries_output"] = summaries_output

            progress(1.0, desc="✅ AI summarization completed!")
            return summaries_output
        else:
            # No transcripts available for summarization
            summaries_output = "❌ No transcripts available for summarization."
            pipeline_state["summaries_output"] = summaries_output
            return summaries_output

    except Exception as e:
        print(f"[ERROR] AI summarization failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return f"❌ AI Summarization Error: {str(e)}"


def step4_generate_comparison(
    summaries_input: str, progress: gr.Progress = gr.Progress()
) -> str:
    """
    Step 4: Generate comparison table.

    Args:
        summaries_input: Previous step output (not used, just for chaining)

    Returns:
        str: Formatted comparison table
    """
    try:
        global pipeline_state

        # Check if we have valid state from previous steps
        if not pipeline_state["pipeline"]:
            return "❌ Error: No valid pipeline state. Please run previous steps first."

        pipeline = pipeline_state["pipeline"]

        # Generate comparison table with progress tracking
        progress(0.1, desc="📊 Generating comparison table...")
        progress(0.3, desc="🤖 Running parallel AI insight generation...")
        comparison_table = generate_comparison_table_with_script(pipeline.output_folder)
        pipeline_state["comparison_table"] = comparison_table

        progress(1.0, desc="✅ Comparison table completed!")
        return comparison_table

    except Exception as e:
        print(f"[ERROR] Comparison table generation failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return f"❌ Comparison Table Error: {str(e)}"


def step5_generate_assignments(
    comparison_input: str, progress: gr.Progress = gr.Progress()
) -> str:
    """
    Step 5: Generate educational assignments.

    Args:
        comparison_input: Previous step output (not used, just for chaining)

    Returns:
        str: Formatted assignments results
    """
    try:
        global pipeline_state

        # Check if we have valid state from previous steps
        if not pipeline_state["pipeline"]:
            return "❌ Error: No valid pipeline state. Please run previous steps first."

        pipeline = pipeline_state["pipeline"]

        # Generate assignments with progress tracking
        progress(0.1, desc="📝 Initializing assignment generator...")

        # Import the assignment generator
        import os
        import sys

        sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
        from assignment_generator import YouTubeAssignmentGenerator

        # Get the current pipeline state to use the same worker configuration
        num_workers = 2  # Default fallback value

        # Use the same number of workers as configured in the pipeline for consistency
        if pipeline_state.get("pipeline") and hasattr(
            pipeline_state["pipeline"], "num_workers"
        ):
            num_workers = max(
                pipeline_state["pipeline"].num_workers, 2
            )  # Minimum 2 workers

        progress(0.3, desc="🤖 Running parallel assignment generation...")

        # Initialize the assignment generator
        generator = YouTubeAssignmentGenerator(
            pipeline_output_folder=pipeline.output_folder, num_workers=num_workers
        )

        progress(0.5, desc="📊 Loading video data and summaries...")

        # Load necessary data
        video_metadata = generator.load_video_metadata()
        summary_data = generator.load_summary_data()

        if not summary_data:
            assignments_output = "❌ No summaries available for assignment generation."
            pipeline_state["assignments_output"] = assignments_output
            return assignments_output

        progress(0.7, desc="🚀 Generating assignments in parallel...")

        # Generate assignments
        assignment_results = generator.generate_assignments(
            video_metadata, summary_data
        )

        # Format the results for display
        assignments_output = format_assignments_results(
            assignment_results,
            video_metadata,
            summary_data,
            generator.assignments_folder,
        )
        pipeline_state["assignments_output"] = assignments_output

        progress(1.0, desc="✅ Assignment generation completed!")
        return assignments_output

    except Exception as e:
        print(f"[ERROR] Assignment generation failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return f"❌ Assignment Generation Error: {str(e)}"


def format_assignments_results(
    assignment_results: Dict[str, bool],
    video_metadata: Dict[str, Dict],
    summary_data: Dict[str, Dict],
    assignments_folder: Path,
) -> str:
    """Format the assignment generation results."""
    if not assignment_results:
        return "❌ No assignments were generated."

    successful_count = sum(assignment_results.values())
    total_count = len(assignment_results)

    result = f"📝 **Educational Assignments Generated** ({successful_count}/{total_count} completed)\n\n"
    result += f"📁 **Assignments saved to:** `{assignments_folder}`\n\n"

    for i, (video_id, success) in enumerate(assignment_results.items(), 1):
        video_info = video_metadata.get(video_id, {})
        title = video_info.get("title", "Unknown Title")
        channel = video_info.get("channel", "Unknown Channel")
        url = video_info.get("url", "#")

        result += f"## {i}. {title}\n"
        result += f"**Channel:** {channel}\n"
        result += f"**URL:** {url}\n\n"

        if success:
            assignment_file = f"{video_id}_assignment.md"
            assignment_path = assignments_folder / assignment_file

            if assignment_path.exists():
                # Read and display the full assignment content
                try:
                    with open(assignment_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Extract the actual assignment content (skip metadata)
                    lines = content.split("\n")
                    assignment_content = []
                    in_metadata = False
                    metadata_blocks = 0

                    for line in lines:
                        if line.strip() == "---":
                            if not in_metadata:
                                in_metadata = True
                                metadata_blocks += 1
                            else:
                                in_metadata = False
                                metadata_blocks += 1
                                if metadata_blocks >= 2:  # Skip the metadata block
                                    continue
                        elif not in_metadata and metadata_blocks >= 2:
                            assignment_content.append(line)

                    # Join the assignment content and clean it up
                    full_assignment = "\n".join(assignment_content).strip()

                    result += f"✅ **Assignment Generated Successfully**\n\n"
                    result += f"### 📝 Full Assignment Content:\n\n"
                    result += full_assignment + "\n\n"

                except Exception as e:
                    result += f"✅ **Assignment Generated** - `{assignment_file}`\n"
                    result += f"⚠️ Error reading assignment content: {str(e)}\n\n"
            else:
                result += f"✅ **Assignment Generated** - `{assignment_file}`\n\n"
        else:
            result += f"❌ **Assignment Generation Failed**\n"
            result += f"Please check the logs for error details.\n\n"

        result += "---\n\n"

    # Add summary statistics
    result += f"## 📊 Generation Summary\n\n"
    result += f"- **Total Videos:** {total_count}\n"
    result += f"- **Successful Assignments:** {successful_count}\n"
    result += f"- **Success Rate:** {successful_count/total_count*100:.1f}%\n"
    result += f"- **Output Folder:** `{assignments_folder}`\n\n"

    result += f"### 🎯 Assignment Features\n\n"
    result += f"Each assignment includes:\n"
    result += (
        f"- **📋 Assignment Overview** - Clear learning objectives and deliverables\n"
    )
    result += f"- **📚 Prerequisite Knowledge** - Required background and skills\n"
    result += f"- **🔧 Core Tasks** - Progressive, hands-on implementation tasks\n"
    result += f"- **💡 Practical Exercises** - Real-world problem-solving scenarios\n"
    result += (
        f"- **🚀 Advanced Challenges** - Extension activities for deeper learning\n"
    )
    result += f"- **✅ Assessment Criteria** - Clear success metrics and rubrics\n"
    result += f"- **📖 Resources & References** - Additional learning materials\n\n"

    return result


def format_transcript_results(
    transcript_paths: List[str], videos: List[Dict], transcripts_folder: str
) -> str:
    """Format the transcript fetching results with full transcript content."""
    if not videos:
        return "❌ No videos to fetch transcripts from."

    result = (
        f"📝 **Video Transcripts** ({len(transcript_paths)}/{len(videos)} fetched)\n\n"
    )

    # Create a mapping of video IDs to transcripts
    transcript_files = {Path(tp).stem.split(".")[0]: tp for tp in transcript_paths}

    for i, video in enumerate(videos, 1):
        title = video.get("title", "Unknown Title")
        video_id = video.get("video_id", "")
        channel = video.get("channel", "Unknown Channel")

        result += f"## {i}. {title}\n"
        result += f"**Channel:** {channel}\n\n"

        if video_id in transcript_files:
            # Load and display full transcript content
            try:
                with open(transcript_files[video_id], "r", encoding="utf-8") as f:
                    transcript_content = f.read()

                # Clean up SRT format and convert to readable text
                import re

                # Remove SRT timestamp lines and sequence numbers
                clean_content = re.sub(
                    r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n",
                    "",
                    transcript_content,
                )
                clean_content = re.sub(r"^\d+$", "", clean_content, flags=re.MULTILINE)
                clean_content = re.sub(r"\n\s*\n", "\n", clean_content)
                clean_content = clean_content.strip()

                if clean_content:
                    result += f"**Full Transcript:**\n\n"
                    # Format transcript as flowing text instead of code block
                    # Split into sentences and create readable paragraphs
                    sentences = clean_content.replace("\n", " ").split(". ")
                    formatted_transcript = ""
                    current_paragraph = ""

                    for i, sentence in enumerate(sentences):
                        sentence = sentence.strip()
                        if sentence:
                            # Add period back if it was removed by split (except for last sentence)
                            if i < len(sentences) - 1 and not sentence.endswith("."):
                                sentence += "."

                            current_paragraph += sentence + " "

                            # Create paragraph breaks every 3-4 sentences for readability
                            if (i + 1) % 4 == 0:
                                formatted_transcript += (
                                    current_paragraph.strip() + "\n\n"
                                )
                                current_paragraph = ""

                    # Add any remaining content
                    if current_paragraph.strip():
                        formatted_transcript += current_paragraph.strip() + "\n\n"

                    result += formatted_transcript
                else:
                    result += f"⚠️ Transcript file exists but appears to be empty or malformed.\n\n"

            except Exception as e:
                result += f"❌ Error reading transcript: {str(e)}\n\n"
        else:
            result += (
                f"❌ **Transcript not available** - Failed to fetch from YouTube\n\n"
            )

        result += "---\n\n"

    return result


def format_summaries_results(
    transcript_paths: List[str], videos: List[Dict], summaries_folder: str
) -> str:
    """Format the AI summarization results with full summary content."""
    if not transcript_paths:
        return "❌ No transcripts available for summarization."

    processed_count = len(transcript_paths)
    successful_count = 0

    # Count successful summaries
    for transcript_path in transcript_paths:
        filename = os.path.basename(transcript_path)
        video_id = filename.split(".")[0]
        summary_filename = f"{video_id}_summary.json"
        summary_path = os.path.join(summaries_folder, summary_filename)
        if os.path.exists(summary_path):
            successful_count += 1

    result = f"🤖 **AI Generated Summaries** ({successful_count}/{processed_count} completed)\n\n"

    for i, transcript_path in enumerate(transcript_paths, 1):
        filename = os.path.basename(transcript_path)
        video_id = filename.split(".")[0]
        video_info = next((v for v in videos if v["video_id"] == video_id), {})
        title = video_info.get("title", "Unknown Title")
        channel = video_info.get("channel", "Unknown Channel")
        url = video_info.get("url", "#")

        summary_filename = f"{video_id}_summary.json"
        summary_path = os.path.join(summaries_folder, summary_filename)

        result += f"## {i}. {title}\n"
        result += f"**Channel:** {channel}\n"
        result += f"**URL:** {url}\n\n"

        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)

                # Handle the new summarizer_v2 JSON structure
                # High-level overview (main summary)
                high_level_overview = summary_data.get("high_level_overview", "")
                if high_level_overview:
                    result += f"### 📋 High-Level Overview\n\n{high_level_overview}\n\n"

                # Technical breakdown
                technical_breakdown = summary_data.get("technical_breakdown", [])
                if technical_breakdown:
                    result += f"### 🔧 Technical Breakdown\n\n"

                    # Group by type for better organization
                    tools = [
                        item
                        for item in technical_breakdown
                        if item.get("type") == "tool"
                    ]
                    architectures = [
                        item
                        for item in technical_breakdown
                        if item.get("type") == "architecture"
                    ]
                    processes = [
                        item
                        for item in technical_breakdown
                        if item.get("type") == "process"
                    ]

                    if tools:
                        result += f"#### 🛠️ Tools & Frameworks\n"
                        for tool in tools:
                            name = tool.get("name", "Unknown Tool")
                            purpose = tool.get("purpose", "Purpose not specified")
                            result += f"• **{name}**: {purpose}\n"
                        result += "\n"

                    if architectures:
                        result += f"#### 🏗️ Architecture & Design\n"
                        for arch in architectures:
                            description = arch.get("description", "No description")
                            result += f"• {description}\n"
                        result += "\n"

                    if processes:
                        result += f"#### 📋 Step-by-Step Process\n"
                        # Sort by step_number
                        processes.sort(key=lambda x: x.get("step_number", 0))
                        for process in processes:
                            step_num = process.get("step_number", "?")
                            description = process.get("description", "No description")
                            result += f"{step_num}. {description}\n"
                        result += "\n"

                # Key insights
                insights = summary_data.get("insights", [])
                if insights:
                    result += f"### 💡 Key Engineering Insights\n\n"
                    for i, insight in enumerate(insights, 1):
                        result += f"{i}. {insight}\n"
                    result += "\n"

                # Practical applications
                applications = summary_data.get("applications", [])
                if applications:
                    result += f"### 🎯 Practical Applications\n\n"
                    for app in applications:
                        result += f"• {app}\n"
                    result += "\n"

                # Limitations and considerations
                limitations = summary_data.get("limitations", [])
                if limitations:
                    result += f"### ⚠️ Limitations & Considerations\n\n"
                    for limitation in limitations:
                        result += f"• {limitation}\n"
                    result += "\n"

            except Exception as e:
                result += f"❌ **Error loading summary:** {str(e)}\n\n"
        else:
            result += f"⏳ **Summary in progress...** 🔄\n\n"

        result += "---\n\n"

    return result


def generate_comparison_table_with_script(pipeline_output_folder: str) -> str:
    """Generate a comparison table using the existing comparison script."""
    try:
        # Import the comparison script
        import os
        import sys

        sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
        from compare_youtube_outputs import YouTubeOutputComparator

        # Get the current pipeline state to use the same worker configuration
        global pipeline_state
        num_workers = 4  # Default fallback value

        # Use the same number of workers as configured in the pipeline for consistency
        if pipeline_state.get("pipeline") and hasattr(
            pipeline_state["pipeline"], "num_workers"
        ):
            num_workers = max(
                pipeline_state["pipeline"].num_workers, 2
            )  # Minimum 2 workers

        # Initialize the comparator with parallel processing optimized for AI insights
        comparator = YouTubeOutputComparator(
            pipeline_output_folder=pipeline_output_folder,
            use_ai_insights=True,  # Enable AI insights for comprehensive comparison
            num_workers=num_workers,  # Use user-configured worker count for better parallel performance
        )

        print(
            f"[INSIGHTS] Using {num_workers} workers for parallel AI insight generation"
        )

        # Run the comparison
        result = comparator.run_comparison(fix_json=True, save_detailed=False)

        # Unpack results (handle different return formats)
        if len(result) >= 6:
            (
                comparison_df,
                insights_report,
                recommendations,
                video_metadata,
                summary_data,
                ai_insights,
            ) = result
        elif len(result) >= 3:
            comparison_df, insights_report, recommendations = result[:3]
        elif len(result) >= 2:
            comparison_df, insights_report = result[:2]
        else:
            return "❌ Error: Could not generate comparison data."

        if comparison_df.empty:
            return "❌ No data available for comparison."

        # Convert DataFrame to HTML table with proper styling
        html_result = f"# 📊 Video Comparison Analysis\n\n"
        html_result += f"**Comparing {len(comparison_df)} videos:**\n\n"

        # Create HTML table with improved formatting and column widths
        html_result += '<div style="overflow-x: auto; background-color: #ffffff; padding: 15px; border-radius: 8px; border: 2px solid #333333; margin: 10px 0; max-width: 100%;">\n'
        html_result += '<table style="width: 100%; min-width: 1400px; border-collapse: collapse; margin: 0; font-size: 13px; background-color: #ffffff; color: #000000; table-layout: fixed;">\n'

        # Table header - include comprehensive columns from the comparison script
        key_columns = [
            "Title",
            "Channel",
            "Published",
            "Difficulty",
            "Teaching Style",
            "Content Depth",
            "Learning Outcome",
            "Target Audience",
            "Prerequisites",
            "Tools Count",
            "Key Technologies",
            "Complexity Score",
        ]
        available_columns = [col for col in key_columns if col in comparison_df.columns]

        # Define column widths for better formatting
        column_widths = {
            "Title": "20%",
            "Channel": "10%",
            "Published": "8%",
            "Difficulty": "8%",
            "Teaching Style": "10%",
            "Content Depth": "8%",
            "Learning Outcome": "15%",
            "Target Audience": "8%",
            "Prerequisites": "10%",
            "Tools Count": "6%",
            "Key Technologies": "12%",
            "Complexity Score": "6%",
        }

        html_result += "<thead>\n"
        html_result += '<tr style="background-color: #ffffff; color: #000000; border-bottom: 3px solid #000000;">\n'
        for col in available_columns:
            icon = {
                "Title": "📺",
                "Channel": "📻",
                "Published": "📅",
                "Difficulty": "📊",
                "Teaching Style": "🎯",
                "Content Depth": "🔍",
                "Learning Outcome": "🎓",
                "Target Audience": "👥",
                "Prerequisites": "📚",
                "Tools Count": "🔧",
                "Key Technologies": "⚙️",
                "Complexity Score": "📈",
            }.get(col, "📋")
            width = column_widths.get(col, "8%")
            html_result += f'<th style="padding: 12px; text-align: left; border: 2px solid #333333; font-weight: bold; color: #000000; font-size: 12px; background-color: #ffffff; width: {width}; word-wrap: break-word;">{icon} {col}</th>\n'
        html_result += "</tr>\n"
        html_result += "</thead>\n"

        # Table body
        html_result += "<tbody>\n"

        for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
            # Alternate row colors with high contrast
            row_color = "#ffffff" if i % 2 == 1 else "#f8f9fa"
            text_color = "#000000"  # Force black text

            html_result += f'<tr style="background-color: {row_color};">\n'

            for col in available_columns:
                value = str(row.get(col, "N/A"))

                # Truncate titles and adjust text based on column
                if col == "Title":
                    value = value[:80] + "..." if len(str(value)) > 80 else value
                elif col == "Learning Outcome":
                    value = value[:100] + "..." if len(str(value)) > 100 else value
                elif col == "Prerequisites":
                    value = value[:80] + "..." if len(str(value)) > 80 else value
                elif col == "Key Technologies":
                    value = value[:60] + "..." if len(str(value)) > 60 else value

                # Style categorical columns with colors
                if col in ["Difficulty", "Content Depth", "Teaching Style"]:
                    color_map = {
                        # Difficulty levels
                        "Beginner": "#28a745",
                        "Intermediate": "#ffc107",
                        "Advanced": "#dc3545",
                        # Content depth
                        "Surface-level": "#ffc107",
                        "Moderate": "#17a2b8",
                        "Deep-dive": "#6f42c1",
                        # Teaching styles
                        "Code-along": "#28a745",
                        "Explanation-heavy": "#17a2b8",
                        "Project-based": "#6f42c1",
                        "Theory-focused": "#fd7e14",
                        "Mixed": "#6c757d",
                        # Default
                        "Unknown": "#6c757d",
                    }
                    bg_color = color_map.get(value, "#6c757d")
                    html_result += f'<td style="padding: 10px; border: 1px solid #333333; text-align: center; background-color: {row_color}; vertical-align: middle;">\n'
                    html_result += f'<span style="background-color: {bg_color}; color: #ffffff; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; white-space: nowrap;">{value}</span>\n'
                    html_result += f"</td>\n"
                elif col in ["Tools Count", "Complexity Score"]:
                    # Special styling for numeric columns
                    html_result += f'<td style="padding: 10px; border: 1px solid #333333; text-align: center; background-color: {row_color}; font-weight: bold; color: #2c5282; font-size: 14px; vertical-align: middle;">{value}</td>\n'
                elif col in [
                    "Title",
                    "Learning Outcome",
                    "Prerequisites",
                    "Key Technologies",
                ]:
                    # Text columns with word wrapping
                    html_result += f'<td style="padding: 10px; border: 1px solid #333333; color: {text_color}; vertical-align: top; background-color: {row_color}; font-size: 12px; line-height: 1.4; word-wrap: break-word; overflow-wrap: break-word;">{value}</td>\n'
                else:
                    # Other columns with standard formatting
                    html_result += f'<td style="padding: 10px; border: 1px solid #333333; color: {text_color}; vertical-align: middle; background-color: {row_color}; font-size: 12px; text-align: center;">{value}</td>\n'

            html_result += "</tr>\n"

        html_result += "</tbody>\n"
        html_result += "</table>\n"
        html_result += "</div>\n\n"

        return html_result

    except Exception as e:
        return f"❌ Error generating comparison: {str(e)}\n\nPlease ensure the pipeline has completed and output files are available."


def format_search_results(videos: List[Dict]) -> str:
    """Format the video search results."""
    if not videos:
        return "❌ No videos found for the search query."

    results = f"🔍 **Found {len(videos)} videos:**\n\n"

    for i, video in enumerate(videos, 1):
        title = video.get("title", "Unknown Title")
        channel = video.get("channel", "Unknown Channel")
        url = video.get("url", "#")
        description = video.get("description", "")
        publish_date = video.get("published_at", "Unknown Date")
        duration = video.get("duration", "Unknown")

        # Truncate description if too long
        if len(description) > 200:
            description = description[:200] + "..."

        results += f"""**{i}. {title}**
   • **Channel:** {channel}
   • **Published:** {publish_date}
   • **Duration:** {duration}
   • **URL:** {url}
   • **Description:** {description}

"""

    return results


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
    
    /* Transcript text formatting for full width */
    .transcript-container textarea {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        line-height: 1.6 !important;
        word-wrap: break-word !important;
        white-space: pre-wrap !important;
        text-align: justify !important;
        width: 100% !important;
    }
    
    /* Ensure textboxes use full width */
    .gradio-textbox {
        width: 100% !important;
    }
    
    .gradio-textbox textarea {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
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

            # Right Column - Start Pipeline Button
            with gr.Column(scale=1, elem_classes=["results-section"]):
                gr.HTML('<h3 class="section-header">🚀 Ready to Start</h3>')

                gr.Markdown(
                    """
                **Pipeline Steps:**
                1. 🔍 Search YouTube videos
                2. 📝 Fetch video transcripts  
                3. 🤖 Generate AI summaries
                4. 📊 Create comparison analysis
                5. 📝 Generate educational assignments
                
                **Sequential Execution:** Each step runs independently with its own progress bar!
                
                ✨ **Key Benefits:**
                - No progress bar duplication
                - Completed outputs remain visible
                - Clear step-by-step progression
                - Individual progress tracking per step
                """
                )

        # Pipeline Results - Real-time output blocks in single column
        gr.HTML(
            '<h2 style="text-align: center; color: #667eea; margin: 30px 0 20px 0;">📊 Pipeline Results</h2>'
        )

        # Single column layout
        search_results = gr.Textbox(
            label="🔍 1. YouTube Search Results",
            lines=8,
            max_lines=15,
            show_copy_button=True,
            info="Found videos with details",
            interactive=False,
        )

        transcripts_output = gr.Textbox(
            label="📝 2. Video Transcripts",
            lines=8,
            max_lines=15,
            show_copy_button=True,
            info="Extracted transcripts with previews",
            interactive=False,
            container=True,
            autoscroll=False,
            elem_classes=["transcript-container"],
        )

        summaries_output = gr.Textbox(
            label="🤖 3. AI Summaries",
            lines=8,
            max_lines=15,
            show_copy_button=True,
            info="AI-generated summaries and key points",
            interactive=False,
        )

        comparison_table = gr.HTML(
            label="📊 4. Video Comparison Analysis",
            value="<div style='padding: 20px; text-align: center; color: #000000; background-color: #ffffff; border: 2px solid #333333; border-radius: 8px; font-weight: bold;'>Comparison table will appear here after all summaries are generated...</div>",
            visible=True,
        )

        assignments_output = gr.Textbox(
            label="📝 5. Educational Assignments",
            lines=8,
            max_lines=15,
            show_copy_button=True,
            info="AI-generated educational assignments for hands-on learning",
            interactive=False,
        )

        # Sequential pipeline execution using .then() method
        # Step 1: Search for videos
        step1_event = process_btn.click(
            fn=step1_search_videos,
            inputs=[
                search_query,
                max_videos,
                transcript_language,
                num_workers,
                openai_api_key,
                youtube_api_key,
                use_env_keys,
            ],
            outputs=search_results,
            show_progress="full",  # Show progress only for this output
        )

        # Step 2: Fetch transcripts (triggered after step 1 completes)
        step2_event = step1_event.then(
            fn=step2_fetch_transcripts,
            inputs=search_results,  # Pass the search results as input (for chaining)
            outputs=transcripts_output,
            show_progress="full",  # Show progress only for this output
        )

        # Step 3: Generate summaries (triggered after step 2 completes)
        step3_event = step2_event.then(
            fn=step3_generate_summaries,
            inputs=transcripts_output,  # Pass the transcript results as input (for chaining)
            outputs=summaries_output,
            show_progress="full",  # Show progress only for this output
        )

        # Step 4: Generate comparison table (triggered after step 3 completes)
        step4_event = step3_event.then(
            fn=step4_generate_comparison,
            inputs=summaries_output,  # Pass the summaries results as input (for chaining)
            outputs=comparison_table,
            show_progress="full",  # Show progress only for this output
        )

        # Step 5: Generate assignments (triggered after step 4 completes)
        step5_event = step4_event.then(
            fn=step5_generate_assignments,
            inputs=comparison_table,  # Pass the comparison results as input (for chaining)
            outputs=assignments_output,
            show_progress="full",  # Show progress only for this output
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
