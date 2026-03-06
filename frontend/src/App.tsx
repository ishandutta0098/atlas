import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowUpRight, Clapperboard, Database, Sparkles } from 'lucide-react'
import { useState } from 'react'

import { Badge } from './components/ui/badge'
import { Card } from './components/ui/card'
import { PapersPanel } from './features/papers/papers-panel'
import { PipelineDashboard } from './features/pipeline/pipeline-dashboard'
import {
  getLatestRun,
  getPapersStatus,
  getRunBundle,
  getRuns,
  queryPapers,
  searchPipeline,
  triggerAssignments,
  triggerComparison,
  triggerSummaries,
  triggerTranscripts,
} from './lib/api'
import type { ArtifactGenerationRequest } from './lib/types'

const defaultArtifactRequest: ArtifactGenerationRequest = {
  refresh: true,
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

  const runsQuery = useQuery({
    queryKey: ['runs'],
    queryFn: getRuns,
  })

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

  const searchMutation = useMutation({
    mutationFn: () =>
      searchPipeline({
        query: searchQuery,
        max_videos: maxVideos,
        transcript_language: transcriptLanguage,
        num_workers: numWorkers,
        use_env_keys: true,
        prefer_cache: false,
      }),
    onSuccess: (result) => {
      setSelectedRunId(result.run_id)
      setStatusMessage(result.detail)
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
      void queryClient.invalidateQueries({ queryKey: ['runs', 'latest'] })
      void queryClient.invalidateQueries({ queryKey: ['runs', result.run_id, 'bundle'] })
    },
    onError: (error) => {
      setStatusMessage(error.message)
    },
  })

  const transcriptsMutation = useMutation({
    mutationFn: () =>
      triggerTranscripts(activeRunId, {
        ...defaultArtifactRequest,
        transcript_language: transcriptLanguage,
        num_workers: numWorkers,
      }),
    onSuccess: (result) => {
      setStatusMessage(result.detail)
      void queryClient.invalidateQueries({ queryKey: ['runs', activeRunId, 'bundle'] })
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
    onError: (error) => setStatusMessage(error.message),
  })

  const summariesMutation = useMutation({
    mutationFn: () =>
      triggerSummaries(activeRunId, {
        ...defaultArtifactRequest,
        transcript_language: transcriptLanguage,
        num_workers: numWorkers,
      }),
    onSuccess: (result) => {
      setStatusMessage(result.detail)
      void queryClient.invalidateQueries({ queryKey: ['runs', activeRunId, 'bundle'] })
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
    onError: (error) => setStatusMessage(error.message),
  })

  const comparisonMutation = useMutation({
    mutationFn: () =>
      triggerComparison(activeRunId, {
        ...defaultArtifactRequest,
        transcript_language: transcriptLanguage,
        num_workers: numWorkers,
        use_ai_insights: false,
      }),
    onSuccess: (result) => {
      setStatusMessage(result.detail)
      void queryClient.invalidateQueries({ queryKey: ['runs', activeRunId, 'bundle'] })
    },
    onError: (error) => setStatusMessage(error.message),
  })

  const assignmentsMutation = useMutation({
    mutationFn: () =>
      triggerAssignments(activeRunId, {
        ...defaultArtifactRequest,
        transcript_language: transcriptLanguage,
        num_workers: numWorkers,
      }),
    onSuccess: (result) => {
      setStatusMessage(result.detail)
      void queryClient.invalidateQueries({ queryKey: ['runs', activeRunId, 'bundle'] })
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
    onError: (error) => setStatusMessage(error.message),
  })

  const papersMutation = useMutation({
    mutationFn: () => queryPapers(papersQuery),
    onError: (error) => setStatusMessage(error.message),
  })

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(56,189,248,0.16),transparent_28%),radial-gradient(circle_at_85%_12%,rgba(168,85,247,0.18),transparent_24%),radial-gradient(circle_at_65%_70%,rgba(34,197,94,0.14),transparent_30%)]" />

      <div className="relative mx-auto max-w-[1480px] px-6 pb-16 pt-10 lg:px-10">
        <header className="mb-10 flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex items-center gap-3">
              <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/10 p-3 text-cyan-100">
                <Clapperboard className="size-6" />
              </div>
              <div>
                <p className="text-sm uppercase tracking-[0.32em] text-cyan-200">Atlas</p>
                <h1 className="mt-1 text-4xl font-semibold tracking-tight">Educational RAG workspace</h1>
              </div>
            </div>
            <p className="mt-4 max-w-3xl text-sm leading-7 text-slate-300">
              FastAPI now serves structured pipeline artifacts, and the React client renders them as a
              polished teaching product built for demos, comparisons, and guided exploration.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <Badge className="border-cyan-400/30 bg-cyan-400/10 text-cyan-100">
              {latestRunQuery.data?.is_fallback ? 'Fallback ready' : 'API powered'}
            </Badge>
            <Badge>{runsQuery.data?.runs.length ?? 0} runs</Badge>
            <Badge className="border-emerald-400/30 bg-emerald-400/10 text-emerald-100">
              {papersStatusQuery.data?.ready ? 'Papers indexed' : 'Papers loading'}
            </Badge>
          </div>
        </header>

        {statusMessage ? (
          <Card className="mb-8 border-cyan-400/20 bg-cyan-400/10 p-4 text-sm text-cyan-50">
            {statusMessage}
          </Card>
        ) : null}

        <div className="mb-8 grid gap-4 lg:grid-cols-3">
          <Card className="p-5">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                <Database className="size-5 text-cyan-200" />
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Cached-first</p>
                <p className="mt-1 text-lg font-semibold text-white">Instant demo surface</p>
              </div>
            </div>
            <p className="mt-4 text-sm leading-7 text-slate-300">
              The app hydrates from the existing fallback run before any live API call is needed.
            </p>
          </Card>
          <Card className="p-5">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                <Sparkles className="size-5 text-violet-200" />
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Structured output</p>
                <p className="mt-1 text-lg font-semibold text-white">No Gradio-only formatting</p>
              </div>
            </div>
            <p className="mt-4 text-sm leading-7 text-slate-300">
              Search, transcripts, summaries, comparison rows, assignments, and papers now flow through
              typed JSON contracts.
            </p>
          </Card>
          <Card className="p-5">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                <ArrowUpRight className="size-5 text-emerald-200" />
              </div>
              <div>
                <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Live capable</p>
                <p className="mt-1 text-lg font-semibold text-white">Fresh runs still supported</p>
              </div>
            </div>
            <p className="mt-4 text-sm leading-7 text-slate-300">
              The same interface can launch new searches and regenerate downstream artifacts when keys
              are available.
            </p>
          </Card>
        </div>

        <main className="space-y-14">
          <PipelineDashboard
            activeRunId={activeRunId}
            bundle={runBundleQuery.data}
            error={runBundleQuery.error instanceof Error ? runBundleQuery.error.message : undefined}
            isGeneratingAssignments={assignmentsMutation.isPending}
            isGeneratingSummaries={summariesMutation.isPending}
            isGeneratingTranscripts={transcriptsMutation.isPending}
            isLoading={runBundleQuery.isLoading || latestRunQuery.isLoading}
            isRefreshingComparison={comparisonMutation.isPending}
            isSearching={searchMutation.isPending}
            maxVideos={maxVideos}
            numWorkers={numWorkers}
            onGenerateAssignments={() => assignmentsMutation.mutate()}
            onGenerateSummaries={() => summariesMutation.mutate()}
            onGenerateTranscripts={() => transcriptsMutation.mutate()}
            onMaxVideosChange={setMaxVideos}
            onNumWorkersChange={setNumWorkers}
            onRefreshComparison={() => comparisonMutation.mutate()}
            onSearchQueryChange={setSearchQuery}
            onSelectRun={setSelectedRunId}
            onStartRun={() => searchMutation.mutate()}
            onTranscriptLanguageChange={setTranscriptLanguage}
            runs={runsQuery.data?.runs ?? []}
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
