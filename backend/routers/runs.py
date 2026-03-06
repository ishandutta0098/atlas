from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.runs import (
    AssignmentArtifactResponse,
    ComparisonArtifactResponse,
    RunListResponse,
    RunManifest,
    SearchArtifactResponse,
    SummaryArtifactResponse,
    TranscriptArtifactResponse,
)
from backend.services.run_service import RunService

router = APIRouter(prefix="/api/runs", tags=["runs"])
run_service = RunService()


@router.get("", response_model=RunListResponse)
def list_runs() -> RunListResponse:
    return RunListResponse(runs=run_service.list_runs())


@router.get("/latest", response_model=RunManifest)
def get_latest_run() -> RunManifest:
    return run_service.get_latest_run()


@router.get("/{run_id}", response_model=RunManifest)
def get_run_manifest(run_id: str) -> RunManifest:
    return run_service.get_manifest(run_id)


@router.get("/{run_id}/videos", response_model=SearchArtifactResponse)
def get_run_videos(run_id: str) -> SearchArtifactResponse:
    return run_service.get_search_artifact(run_id)


@router.get("/{run_id}/transcripts", response_model=TranscriptArtifactResponse)
def get_run_transcripts(run_id: str) -> TranscriptArtifactResponse:
    return run_service.get_transcripts(run_id)


@router.get("/{run_id}/summaries", response_model=SummaryArtifactResponse)
def get_run_summaries(run_id: str) -> SummaryArtifactResponse:
    return run_service.get_summaries(run_id)


@router.get("/{run_id}/comparison", response_model=ComparisonArtifactResponse)
def get_run_comparison(run_id: str) -> ComparisonArtifactResponse:
    return run_service.get_comparison(run_id)


@router.get("/{run_id}/assignments", response_model=AssignmentArtifactResponse)
def get_run_assignments(run_id: str) -> AssignmentArtifactResponse:
    return run_service.get_assignments(run_id)
