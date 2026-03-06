from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.pipeline import ArtifactGenerationRequest, PipelineActionResponse, SearchRequest
from backend.services.pipeline_service import PipelineService

router = APIRouter(prefix="/api", tags=["pipeline"])
pipeline_service = PipelineService()


@router.post("/pipeline/search", response_model=PipelineActionResponse)
def search_pipeline(request: SearchRequest) -> PipelineActionResponse:
    return pipeline_service.search(request)


@router.post("/runs/{run_id}/transcripts", response_model=PipelineActionResponse)
def generate_transcripts(run_id: str, request: ArtifactGenerationRequest) -> PipelineActionResponse:
    return pipeline_service.generate_transcripts(run_id, request)


@router.post("/runs/{run_id}/summaries", response_model=PipelineActionResponse)
def generate_summaries(run_id: str, request: ArtifactGenerationRequest) -> PipelineActionResponse:
    return pipeline_service.generate_summaries(run_id, request)


@router.post("/runs/{run_id}/comparison", response_model=PipelineActionResponse)
def generate_comparison(run_id: str, request: ArtifactGenerationRequest) -> PipelineActionResponse:
    return pipeline_service.generate_comparison(run_id, request)


@router.post("/runs/{run_id}/assignments", response_model=PipelineActionResponse)
def generate_assignments(run_id: str, request: ArtifactGenerationRequest) -> PipelineActionResponse:
    return pipeline_service.generate_assignments(run_id, request)
