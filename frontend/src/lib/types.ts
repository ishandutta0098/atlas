export interface ArtifactAvailability {
  videos: boolean
  transcripts: boolean
  summaries: boolean
  comparison: boolean
  assignments: boolean
}

export interface ArtifactCounts {
  videos: number
  transcripts: number
  summaries: number
  assignments: number
}

export interface RunManifest {
  run_id: string
  source_folder: string
  search_query: string
  created_at: string
  updated_at: number
  is_fallback: boolean
  is_demo_ready: boolean
  availability: ArtifactAvailability
  counts: ArtifactCounts
}

export interface VideoResult {
  title: string
  channel: string
  url: string
  description: string
  published_at: string
  video_id: string
  duration: string
}

export interface SearchArtifactResponse {
  run: RunManifest
  search_query: string
  timestamp: string
  total_videos_found: number
  max_videos_requested: number
  videos: VideoResult[]
}

export interface TranscriptArtifact {
  video_id: string
  title: string
  channel: string
  language: string
  transcript_path: string | null
  raw_srt: string
  cleaned_text: string
  available: boolean
}

export interface TranscriptArtifactResponse {
  run: RunManifest
  items: TranscriptArtifact[]
}

export interface SummaryArtifact {
  video_id: string
  title: string
  channel: string
  url: string
  summary_path: string | null
  high_level_overview: string
  technical_breakdown: Array<Record<string, unknown>>
  insights: string[]
  applications: string[]
  limitations: string[]
  available: boolean
}

export interface SummaryArtifactResponse {
  run: RunManifest
  items: SummaryArtifact[]
}

export interface ComparisonRow {
  video_id: string
  title: string
  channel: string
  published: string
  recency: string
  difficulty: string
  teaching_style: string
  practical_value: string
  content_depth: string
  worth_time: string
  learning_outcome: string
  target_audience: string
  prerequisites: string
  key_differentiators: string
  tools_count: number
  key_technologies: string[]
  complexity_score: number
  summary_available: boolean
  url: string
  full_overview: string
  insights: string[]
  applications: string[]
  limitations: string[]
}

export interface ComparisonArtifactResponse {
  run: RunManifest
  rows: ComparisonRow[]
  insights_report: string
  recommendations: string[]
  used_ai_insights: boolean
}

export interface AssignmentArtifact {
  video_id: string
  title: string
  channel: string
  url: string
  assignment_path: string | null
  metadata: Record<string, unknown>
  markdown: string
  available: boolean
}

export interface AssignmentArtifactResponse {
  run: RunManifest
  items: AssignmentArtifact[]
}

export interface RunBundle {
  videos: SearchArtifactResponse
  transcripts: TranscriptArtifactResponse
  summaries: SummaryArtifactResponse
  comparison: ComparisonArtifactResponse
  assignments: AssignmentArtifactResponse
}

export interface PapersStatusResponse {
  ready: boolean
  initialized: boolean
  papers_folder: string
  pdf_count: number
  index_exists: boolean
  vector_store_exists: boolean
  message: string
}

export interface PapersSource {
  file_name: string
  title: string
  authors: string
  score: number
  text: string
  page_count: number
  has_abstract: boolean
}

export interface PapersQueryResponse {
  success: boolean
  query: string
  response: string
  sources: PapersSource[]
  search_time: number
  num_sources: number
  error: string
}

export interface SearchRequest {
  query: string
  max_videos: number
  transcript_language: string
  num_workers: number
  use_env_keys: boolean
  openrouter_api_key?: string
  youtube_api_key?: string
  prefer_cache: boolean
}

export interface ArtifactGenerationRequest {
  refresh: boolean
  num_workers?: number
  transcript_language: string
  use_ai_insights: boolean
}

export interface PipelineActionResponse {
  run_id: string
  source_folder: string
  status: string
  detail: string
}
