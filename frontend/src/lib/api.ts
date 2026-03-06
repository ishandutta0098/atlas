import type {
  ArtifactGenerationRequest,
  PapersQueryResponse,
  PapersStatusResponse,
  PipelineActionResponse,
  RunBundle,
  RunManifest,
  SearchRequest,
} from './types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })

  if (!response.ok) {
    let message = 'Request failed.'
    try {
      const errorBody = (await response.json()) as { detail?: string }
      message = errorBody.detail ?? message
    } catch {
      message = response.statusText || message
    }
    throw new Error(message)
  }

  return (await response.json()) as T
}

export function getRuns() {
  return request<{ runs: RunManifest[] }>('/api/runs')
}

export function getLatestRun() {
  return request<RunManifest>('/api/runs/latest')
}

export async function getRunBundle(runId: string): Promise<RunBundle> {
  const [videos, transcripts, summaries, comparison, assignments] = await Promise.all([
    request<RunBundle['videos']>(`/api/runs/${runId}/videos`),
    request<RunBundle['transcripts']>(`/api/runs/${runId}/transcripts`),
    request<RunBundle['summaries']>(`/api/runs/${runId}/summaries`),
    request<RunBundle['comparison']>(`/api/runs/${runId}/comparison`),
    request<RunBundle['assignments']>(`/api/runs/${runId}/assignments`),
  ])

  return {
    videos,
    transcripts,
    summaries,
    comparison,
    assignments,
  }
}

export function searchPipeline(payload: SearchRequest) {
  return request<PipelineActionResponse>('/api/pipeline/search', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function triggerTranscripts(runId: string, payload: ArtifactGenerationRequest) {
  return request<PipelineActionResponse>(`/api/runs/${runId}/transcripts`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function triggerSummaries(runId: string, payload: ArtifactGenerationRequest) {
  return request<PipelineActionResponse>(`/api/runs/${runId}/summaries`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function triggerComparison(runId: string, payload: ArtifactGenerationRequest) {
  return request<PipelineActionResponse>(`/api/runs/${runId}/comparison`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function triggerAssignments(runId: string, payload: ArtifactGenerationRequest) {
  return request<PipelineActionResponse>(`/api/runs/${runId}/assignments`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function getPapersStatus() {
  return request<PapersStatusResponse>('/api/papers/status')
}

export function queryPapers(query: string) {
  return request<PapersQueryResponse>('/api/papers/query', {
    method: 'POST',
    body: JSON.stringify({ query }),
  })
}
