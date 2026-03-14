import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { Card } from './components/ui/card'
import { PapersPanel } from './features/papers/papers-panel'
import { PipelineDashboard } from './features/pipeline/pipeline-dashboard'
import {
  getLatestRun,
  getPapersStatus,
  getRunBundle,
  queryPapers,
  searchPipeline,
  triggerAssignments,
  triggerComparison,
  triggerSummaries,
  triggerTranscripts,
} from './lib/api'
import type { ArtifactGenerationRequest } from './lib/types'

const defaultArtifactRequest: ArtifactGenerationRequest = {
  refresh: false,
  transcript_language: 'en',
  use_ai_insights: false,
}

function App() {
  const queryClient = useQueryClient()
  const [selectedRunId, setSelectedRunId] = useState('')
  const [statusMessage, setStatusMessage] = useState('')
  const [searchQuery, setSearchQuery] = useState('CrewAI Tutorial')
  const [maxVideos, setMaxVideos] = useState(4)
  const [transcriptLanguage, setTranscriptLanguage] = useState('en')
  const [numWorkers, setNumWorkers] = useState(4)
  const [papersQuery, setPapersQuery] = useState('What are the main architectures for AI agents?')

  const latestRunQuery = useQuery({
    queryKey: ['runs', 'latest'],
    queryFn: getLatestRun,
  })

  const activeRunId = selectedRunId || latestRunQuery.data?.run_id || ''

  const runBundleQuery = useQuery({
    enabled: Boolean(activeRunId),
    queryKey: ['runs', activeRunId, 'bundle'],
    queryFn: () => getRunBundle(activeRunId),
  })

  const papersStatusQuery = useQuery({
    queryKey: ['papers', 'status'],
    queryFn: getPapersStatus,
  })

  const runMutation = useMutation({
    mutationFn: async () => {
      const result = await searchPipeline({
        query: searchQuery,
        max_videos: maxVideos,
        transcript_language: transcriptLanguage,
        num_workers: numWorkers,
        use_env_keys: true,
        prefer_cache: false,
      })

      await triggerTranscripts(result.run_id, {
        ...defaultArtifactRequest,
        transcript_language: transcriptLanguage,
        num_workers: numWorkers,
      })

      await triggerSummaries(result.run_id, {
        ...defaultArtifactRequest,
        transcript_language: transcriptLanguage,
        num_workers: numWorkers,
      })

      await Promise.all([
        triggerComparison(result.run_id, {
          ...defaultArtifactRequest,
          transcript_language: transcriptLanguage,
          num_workers: numWorkers,
          use_ai_insights: false,
        }),
        triggerAssignments(result.run_id, {
          ...defaultArtifactRequest,
          transcript_language: transcriptLanguage,
          num_workers: numWorkers,
        }),
      ])

      return result
    },
    onSuccess: (result) => {
      setSelectedRunId(result.run_id)
      setStatusMessage('Run ready.')
      void queryClient.invalidateQueries({ queryKey: ['runs', 'latest'] })
      void queryClient.invalidateQueries({ queryKey: ['runs', result.run_id, 'bundle'] })
    },
    onError: (error) => {
      setStatusMessage(error.message)
    },
  })

  const papersMutation = useMutation({
    mutationFn: () => queryPapers(papersQuery),
    onError: (error) => setStatusMessage(error.message),
  })

  return (
    <div className="min-h-screen bg-[#0b0d10] text-white">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.04),transparent_28%),radial-gradient(circle_at_20%_20%,rgba(111,118,130,0.08),transparent_24%),radial-gradient(circle_at_80%_0%,rgba(205,151,74,0.06),transparent_22%)]" />

      <div className="relative mx-auto max-w-[1380px] px-6 pb-16 pt-10 lg:px-10">
        <header className="mb-10 border-b border-white/6 pb-6 text-center">
          <h1 className="text-4xl font-semibold tracking-[-0.06em] text-white md:text-5xl">Atlas</h1>
        </header>

        {statusMessage ? (
          <Card className="mb-6 border-white/8 bg-white/[0.03] px-4 py-3 text-sm text-zinc-200">
            {statusMessage}
          </Card>
        ) : null}

        <main className="space-y-12">
          <PipelineDashboard
            activeRunId={activeRunId}
            bundle={runBundleQuery.data}
            error={runBundleQuery.error instanceof Error ? runBundleQuery.error.message : undefined}
            isLoading={runBundleQuery.isLoading || latestRunQuery.isLoading}
            isRunning={runMutation.isPending}
            maxVideos={maxVideos}
            numWorkers={numWorkers}
            onMaxVideosChange={setMaxVideos}
            onSearchQueryChange={setSearchQuery}
            onStartRun={() => runMutation.mutate()}
            onTranscriptLanguageChange={setTranscriptLanguage}
            onNumWorkersChange={setNumWorkers}
            searchQuery={searchQuery}
            transcriptLanguage={transcriptLanguage}
          />

          <PapersPanel
            isLoading={papersMutation.isPending}
            onQueryChange={setPapersQuery}
            onSubmit={() => papersMutation.mutate()}
            query={papersQuery}
            result={papersMutation.data}
            status={papersStatusQuery.data}
          />
        </main>
      </div>
    </div>
  )
}

export default App
