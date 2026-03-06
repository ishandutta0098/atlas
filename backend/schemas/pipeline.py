from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    max_videos: int = Field(default=4, ge=1, le=10)
    transcript_language: str = "en"
    num_workers: int = Field(default=4, ge=0, le=16)
    use_env_keys: bool = True
    openrouter_api_key: Optional[str] = None
    youtube_api_key: Optional[str] = None
    prefer_cache: bool = True


class ArtifactGenerationRequest(BaseModel):
    refresh: bool = False
    num_workers: Optional[int] = Field(default=None, ge=0, le=16)
    transcript_language: str = "en"
    use_ai_insights: bool = False


class PipelineActionResponse(BaseModel):
    run_id: str
    source_folder: str
    status: str
    detail: str
