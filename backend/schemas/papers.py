from __future__ import annotations

from pydantic import BaseModel, Field


class PapersStatusResponse(BaseModel):
    ready: bool
    initialized: bool
    papers_folder: str
    pdf_count: int
    index_exists: bool
    vector_store_exists: bool
    message: str = ""


class PapersQueryRequest(BaseModel):
    query: str


class PapersSource(BaseModel):
    file_name: str = ""
    title: str = ""
    authors: str = ""
    score: float = 0
    text: str = ""
    page_count: int = 0
    has_abstract: bool = False


class PapersQueryResponse(BaseModel):
    success: bool
    query: str = ""
    response: str = ""
    sources: list[PapersSource] = Field(default_factory=list)
    search_time: float = 0
    num_sources: int = 0
    error: str = ""
