from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ArtifactAvailability(BaseModel):
    videos: bool = False
    transcripts: bool = False
    summaries: bool = False
    comparison: bool = False
    assignments: bool = False


class ArtifactCounts(BaseModel):
    videos: int = 0
    transcripts: int = 0
    summaries: int = 0
    assignments: int = 0


class RunManifest(BaseModel):
    run_id: str
    source_folder: str
    search_query: str = ""
    created_at: str = ""
    updated_at: float = 0
    is_fallback: bool = False
    is_demo_ready: bool = False
    availability: ArtifactAvailability = Field(default_factory=ArtifactAvailability)
    counts: ArtifactCounts = Field(default_factory=ArtifactCounts)


class RunListResponse(BaseModel):
    runs: list[RunManifest]


class VideoResult(BaseModel):
    title: str
    channel: str
    url: str
    description: str = ""
    published_at: str = ""
    video_id: str
    duration: str = "Unknown"


class SearchArtifactResponse(BaseModel):
    run: RunManifest
    search_query: str = ""
    timestamp: str = ""
    total_videos_found: int = 0
    max_videos_requested: int = 0
    videos: list[VideoResult] = Field(default_factory=list)


class TranscriptArtifact(BaseModel):
    video_id: str
    title: str
    channel: str
    language: str = "en"
    transcript_path: Optional[str] = None
    raw_srt: str = ""
    cleaned_text: str = ""
    available: bool = False


class TranscriptArtifactResponse(BaseModel):
    run: RunManifest
    items: list[TranscriptArtifact] = Field(default_factory=list)


class SummaryArtifact(BaseModel):
    video_id: str
    title: str
    channel: str
    url: str = ""
    summary_path: Optional[str] = None
    high_level_overview: str = ""
    technical_breakdown: list[dict] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    applications: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    available: bool = False


class SummaryArtifactResponse(BaseModel):
    run: RunManifest
    items: list[SummaryArtifact] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    video_id: str
    title: str
    channel: str
    published: str
    recency: str
    difficulty: str
    teaching_style: str
    practical_value: str
    content_depth: str
    worth_time: str
    learning_outcome: str
    target_audience: str
    prerequisites: str
    key_differentiators: str
    tools_count: int
    key_technologies: list[str] = Field(default_factory=list)
    complexity_score: float
    summary_available: bool = False
    url: str = ""
    full_overview: str = ""
    insights: list[str] = Field(default_factory=list)
    applications: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ComparisonArtifactResponse(BaseModel):
    run: RunManifest
    rows: list[ComparisonRow] = Field(default_factory=list)
    insights_report: str = ""
    recommendations: list[str] = Field(default_factory=list)
    used_ai_insights: bool = False


class AssignmentArtifact(BaseModel):
    video_id: str
    title: str
    channel: str
    url: str = ""
    assignment_path: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    display_metadata: dict[str, str] = Field(default_factory=dict)
    markdown: str = ""
    sections: list[dict] = Field(default_factory=list)
    checklist: list[dict] = Field(default_factory=list)
    available: bool = False


class AssignmentArtifactResponse(BaseModel):
    run: RunManifest
    items: list[AssignmentArtifact] = Field(default_factory=list)
