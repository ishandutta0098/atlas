from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.papers import PapersQueryRequest, PapersQueryResponse, PapersStatusResponse
from backend.services.papers_service import PapersService

router = APIRouter(prefix="/api/papers", tags=["papers"])
papers_service = PapersService()


@router.get("/status", response_model=PapersStatusResponse)
def get_papers_status() -> PapersStatusResponse:
    return papers_service.get_status()


@router.post("/query", response_model=PapersQueryResponse)
def query_papers(request: PapersQueryRequest) -> PapersQueryResponse:
    return papers_service.query(request.query)
