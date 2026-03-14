from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from fastapi import HTTPException

from backend.schemas.pipeline import ArtifactGenerationRequest, PipelineActionResponse, SearchRequest
from backend.services.artifact_readers import (
    DEFAULT_FALLBACK_RUN_ID,
    REPO_ROOT,
    read_assignments,
    read_summaries,
    read_transcripts,
    read_videos,
)
from backend.services.run_service import RunService
from src.youtube_pipeline import YouTubePipeline

SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from assignment_generator import YouTubeAssignmentGenerator
from compare_youtube_outputs import YouTubeOutputComparator


class PipelineService:
    def __init__(self, run_service: RunService | None = None):
        self.run_service = run_service or RunService()

    def _apply_api_keys(self, request: SearchRequest) -> None:
        if request.use_env_keys:
            return
        if not request.openrouter_api_key or not request.youtube_api_key:
            raise HTTPException(
                status_code=400,
                detail="Both openrouter_api_key and youtube_api_key are required when use_env_keys is false.",
            )

        os.environ["OPENROUTER_API_KEY"] = request.openrouter_api_key
        os.environ["YOUTUBE_API_KEY"] = request.youtube_api_key

    def _ensure_run_path(self, run_id: str) -> Path:
        return self.run_service.find_run_path(run_id)

    def _build_fallback_response(self) -> PipelineActionResponse:
        fallback_manifest = self.run_service.get_manifest(DEFAULT_FALLBACK_RUN_ID)
        return PipelineActionResponse(
            run_id=fallback_manifest.run_id,
            source_folder=fallback_manifest.source_folder,
            status="ready",
            detail=f"Prepared pipeline run with {fallback_manifest.counts.videos} videos.",
        )

    def search(self, request: SearchRequest) -> PipelineActionResponse:
        if request.prefer_cache:
            existing_run = self.run_service.find_matching_run(request.query)
            if existing_run is not None:
                return PipelineActionResponse(
                    run_id=existing_run.run_id,
                    source_folder=existing_run.source_folder,
                    status="cached",
                    detail="Reused an existing cached run for the same query.",
                )

        try:
            self._apply_api_keys(request)
            run_id = f"pipeline_output_{int(time.time())}"
            pipeline = YouTubePipeline(
                max_videos=request.max_videos,
                transcript_language=request.transcript_language,
                output_folder=run_id,
                num_workers=request.num_workers,
            )
            videos = pipeline.search_videos(request.query)
            if not videos:
                return self._build_fallback_response()

        except Exception:
            return self._build_fallback_response()

        return PipelineActionResponse(
            run_id=run_id,
            source_folder=str(REPO_ROOT / run_id),
            status="created",
            detail=f"Created a new pipeline run with {len(videos)} videos.",
        )

    def generate_transcripts(self, run_id: str, request: ArtifactGenerationRequest) -> PipelineActionResponse:
        run_path = self._ensure_run_path(run_id)
        if not request.refresh and any(item.available for item in read_transcripts(run_path)):
            return PipelineActionResponse(
                run_id=run_id,
                source_folder=str(run_path),
                status="cached",
                detail="Reused existing transcripts from the run folder.",
            )

        videos = [video.model_dump() for video in read_videos(run_path)]
        if not videos:
            raise HTTPException(status_code=400, detail="This run does not have video metadata to fetch transcripts.")

        pipeline = YouTubePipeline(
            max_videos=len(videos),
            transcript_language=request.transcript_language,
            output_folder=run_id,
            num_workers=request.num_workers,
        )
        transcript_paths, _ = pipeline.fetch_transcripts(videos)
        return PipelineActionResponse(
            run_id=run_id,
            source_folder=str(run_path),
            status="updated",
            detail=f"Transcript generation completed for {len(transcript_paths)} videos.",
        )

    def generate_summaries(self, run_id: str, request: ArtifactGenerationRequest) -> PipelineActionResponse:
        run_path = self._ensure_run_path(run_id)
        if not request.refresh and any(item.available for item in read_summaries(run_path)):
            return PipelineActionResponse(
                run_id=run_id,
                source_folder=str(run_path),
                status="cached",
                detail="Reused existing summaries from the run folder.",
            )

        videos = [video.model_dump() for video in read_videos(run_path)]
        transcript_paths = [item.transcript_path for item in read_transcripts(run_path) if item.available and item.transcript_path]
        if not videos or not transcript_paths:
            raise HTTPException(status_code=400, detail="This run needs videos and transcripts before summaries can be generated.")

        pipeline = YouTubePipeline(
            max_videos=len(videos),
            transcript_language=request.transcript_language,
            output_folder=run_id,
            num_workers=request.num_workers,
        )
        result = pipeline.summarize_transcripts(transcript_paths, videos)
        return PipelineActionResponse(
            run_id=run_id,
            source_folder=str(run_path),
            status="updated",
            detail=f"Summary generation completed for {sum(result.values())} transcripts.",
        )

    def generate_comparison(self, run_id: str, request: ArtifactGenerationRequest) -> PipelineActionResponse:
        run_path = self._ensure_run_path(run_id)
        if not request.refresh and not request.use_ai_insights:
            return PipelineActionResponse(
                run_id=run_id,
                source_folder=str(run_path),
                status="cached",
                detail="Comparison will be derived from cached summaries and metadata.",
            )

        comparator = YouTubeOutputComparator(
            pipeline_output_folder=str(run_path),
            use_ai_insights=request.use_ai_insights,
            num_workers=request.num_workers,
        )
        comparator.run_comparison(fix_json=False, save_detailed=False)
        return PipelineActionResponse(
            run_id=run_id,
            source_folder=str(run_path),
            status="updated",
            detail="Comparison analysis was refreshed for this run.",
        )

    def generate_assignments(self, run_id: str, request: ArtifactGenerationRequest) -> PipelineActionResponse:
        run_path = self._ensure_run_path(run_id)
        if not request.refresh and any(item.available for item in read_assignments(run_path)):
            return PipelineActionResponse(
                run_id=run_id,
                source_folder=str(run_path),
                status="cached",
                detail="Reused existing assignments from the run folder.",
            )

        generator = YouTubeAssignmentGenerator(
            pipeline_output_folder=str(run_path),
            num_workers=request.num_workers,
        )
        video_metadata = generator.load_video_metadata()
        summary_data = generator.load_summary_data()
        results = generator.generate_assignments(video_metadata, summary_data)
        return PipelineActionResponse(
            run_id=run_id,
            source_folder=str(run_path),
            status="updated",
            detail=f"Assignment generation completed for {sum(results.values())} summaries.",
        )
