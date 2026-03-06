from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException

from backend.schemas.runs import (
    ArtifactAvailability,
    ArtifactCounts,
    RunManifest,
    SearchArtifactResponse,
    TranscriptArtifactResponse,
    SummaryArtifactResponse,
    ComparisonArtifactResponse,
    AssignmentArtifactResponse,
)
from backend.services.artifact_readers import (
    DEFAULT_FALLBACK_RUN_ID,
    REPO_ROOT,
    build_comparison_artifact,
    build_run_counts,
    detect_run_created_at,
    has_comparison_source_data,
    is_valid_run_directory,
    read_assignments,
    read_search_metadata,
    read_summaries,
    read_transcripts,
    read_videos,
)


class RunService:
    def __init__(self, repo_root: Path | None = None):
        self.repo_root = repo_root or REPO_ROOT

    def list_run_paths(self) -> list[Path]:
        run_paths = [
            path
            for path in self.repo_root.iterdir()
            if re.fullmatch(r"pipeline_output_\d+", path.name) and is_valid_run_directory(path)
        ]
        return sorted(run_paths, key=lambda path: path.stat().st_mtime, reverse=True)

    def find_run_path(self, run_id: str) -> Path:
        run_path = self.repo_root / run_id
        if not is_valid_run_directory(run_path):
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        return run_path

    def get_run_manifest(self, run_path: Path) -> RunManifest:
        counts = build_run_counts(run_path)
        availability = ArtifactAvailability(
            videos=counts["videos"] > 0,
            transcripts=counts["transcripts"] > 0,
            summaries=counts["summaries"] > 0,
            comparison=has_comparison_source_data(run_path),
            assignments=counts["assignments"] > 0,
        )
        search_metadata = read_search_metadata(run_path)
        return RunManifest(
            run_id=run_path.name,
            source_folder=str(run_path),
            search_query=search_metadata.get("search_query", ""),
            created_at=detect_run_created_at(run_path),
            updated_at=run_path.stat().st_mtime,
            is_fallback=run_path.name == DEFAULT_FALLBACK_RUN_ID,
            is_demo_ready=availability.videos and availability.summaries and availability.assignments,
            availability=availability,
            counts=ArtifactCounts(**counts),
        )

    def list_runs(self) -> list[RunManifest]:
        return [self.get_run_manifest(run_path) for run_path in self.list_run_paths()]

    def get_latest_run(self) -> RunManifest:
        fallback_path = self.repo_root / DEFAULT_FALLBACK_RUN_ID
        if is_valid_run_directory(fallback_path):
            return self.get_run_manifest(fallback_path)

        run_paths = self.list_run_paths()
        if not run_paths:
            raise HTTPException(status_code=404, detail="No pipeline runs found.")
        return self.get_run_manifest(run_paths[0])

    def find_matching_run(self, query: str) -> RunManifest | None:
        normalized_query = query.strip().lower()
        for manifest in self.list_runs():
            if manifest.search_query.strip().lower() == normalized_query:
                return manifest
        return None

    def get_manifest(self, run_id: str) -> RunManifest:
        return self.get_run_manifest(self.find_run_path(run_id))

    def get_search_artifact(self, run_id: str) -> SearchArtifactResponse:
        run_path = self.find_run_path(run_id)
        metadata = read_search_metadata(run_path)
        return SearchArtifactResponse(
            run=self.get_run_manifest(run_path),
            search_query=metadata.get("search_query", ""),
            timestamp=metadata.get("timestamp", ""),
            total_videos_found=metadata.get("total_videos_found", 0),
            max_videos_requested=metadata.get("max_videos_requested", 0),
            videos=read_videos(run_path),
        )

    def get_transcripts(self, run_id: str) -> TranscriptArtifactResponse:
        run_path = self.find_run_path(run_id)
        return TranscriptArtifactResponse(
            run=self.get_run_manifest(run_path),
            items=read_transcripts(run_path),
        )

    def get_summaries(self, run_id: str) -> SummaryArtifactResponse:
        run_path = self.find_run_path(run_id)
        return SummaryArtifactResponse(
            run=self.get_run_manifest(run_path),
            items=read_summaries(run_path),
        )

    def get_comparison(self, run_id: str) -> ComparisonArtifactResponse:
        run_path = self.find_run_path(run_id)
        rows, insights_report, recommendations = build_comparison_artifact(run_path)
        return ComparisonArtifactResponse(
            run=self.get_run_manifest(run_path),
            rows=rows,
            insights_report=insights_report,
            recommendations=recommendations,
            used_ai_insights=False,
        )

    def get_assignments(self, run_id: str) -> AssignmentArtifactResponse:
        run_path = self.find_run_path(run_id)
        return AssignmentArtifactResponse(
            run=self.get_run_manifest(run_path),
            items=read_assignments(run_path),
        )
