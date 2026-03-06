import * as Tabs from '@radix-ui/react-tabs'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Bot,
  FileSearch,
  GraduationCap,
  Layers3,
  PlayCircle,
  RefreshCcw,
  Search,
  Sparkles,
  Table2,
  Workflow,
} from 'lucide-react'
import { type ChangeEvent, type FormEvent, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'

import { Badge } from '../../components/ui/badge'
import { Button } from '../../components/ui/button'
import { Card } from '../../components/ui/card'
import type { RunBundle, RunManifest } from '../../lib/types'
import { cn } from '../../lib/utils'

interface PipelineDashboardProps {
  runs: RunManifest[]
  activeRunId: string
  bundle?: RunBundle
  isLoading: boolean
  error?: string
  searchQuery: string
  maxVideos: number
  transcriptLanguage: string
  numWorkers: number
  onSearchQueryChange: (value: string) => void
  onMaxVideosChange: (value: number) => void
  onTranscriptLanguageChange: (value: string) => void
  onNumWorkersChange: (value: number) => void
  onSelectRun: (runId: string) => void
  onStartRun: () => void
  onGenerateTranscripts: () => void
  onGenerateSummaries: () => void
  onRefreshComparison: () => void
  onGenerateAssignments: () => void
  isSearching: boolean
  isGeneratingTranscripts: boolean
  isGeneratingSummaries: boolean
  isRefreshingComparison: boolean
  isGeneratingAssignments: boolean
}

function MetricCard({
  label,
  value,
  icon: Icon,
}: {
  label: string
  value: string | number
  icon: typeof Search
}) {
  return (
    <Card className="p-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-slate-400">{label}</p>
          <p className="mt-2 text-2xl font-semibold text-white">{value}</p>
        </div>
        <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/10 p-3 text-cyan-200">
          <Icon className="size-5" />
        </div>
      </div>
    </Card>
  )
}

function SectionHeading({
  title,
  description,
  action,
}: {
  title: string
  description: string
  action?: ReactNode
}) {
  return (
    <div className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
      <div>
        <h3 className="text-xl font-semibold text-white">{title}</h3>
        <p className="mt-1 text-sm text-slate-400">{description}</p>
      </div>
      {action}
    </div>
  )
}

function formatDate(value: string) {
  if (!value) return 'Unavailable'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

function renderBreakdownItem(item: Record<string, unknown>, index: number) {
  const type = String(item.type ?? 'detail')
  if (type === 'tool') {
    return (
      <li key={`tool-${index}`} className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
        <p className="font-medium text-white">{String(item.name ?? 'Tool')}</p>
        <p className="mt-2 text-sm text-slate-300">{String(item.purpose ?? '')}</p>
      </li>
    )
  }

  if (type === 'process') {
    return (
      <li key={`process-${index}`} className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
        <p className="font-medium text-white">Step {String(item.step_number ?? '?')}</p>
        <p className="mt-2 text-sm text-slate-300">{String(item.description ?? '')}</p>
      </li>
    )
  }

  return (
    <li key={`arch-${index}`} className="rounded-2xl border border-white/10 bg-slate-950/40 p-4">
      <p className="font-medium text-white">Architecture note</p>
      <p className="mt-2 text-sm text-slate-300">{String(item.description ?? '')}</p>
    </li>
  )
}

export function PipelineDashboard({
  runs,
  activeRunId,
  bundle,
  isLoading,
  error,
  searchQuery,
  maxVideos,
  transcriptLanguage,
  numWorkers,
  onSearchQueryChange,
  onMaxVideosChange,
  onTranscriptLanguageChange,
  onNumWorkersChange,
  onSelectRun,
  onStartRun,
  onGenerateTranscripts,
  onGenerateSummaries,
  onRefreshComparison,
  onGenerateAssignments,
  isSearching,
  isGeneratingTranscripts,
  isGeneratingSummaries,
  isRefreshingComparison,
  isGeneratingAssignments,
}: PipelineDashboardProps) {
  const currentRun = bundle?.videos.run
  const videos = bundle?.videos.videos ?? []
  const transcripts = bundle?.transcripts.items ?? []
  const summaries = bundle?.summaries.items ?? []
  const comparison = bundle?.comparison.rows ?? []
  const assignments = bundle?.assignments.items ?? []

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    onStartRun()
  }

  return (
    <section id="pipeline" className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
      >
        <Card className="overflow-hidden">
          <div className="grid gap-10 bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.18),_transparent_42%),radial-gradient(circle_at_top_right,_rgba(168,85,247,0.2),_transparent_38%),linear-gradient(180deg,_rgba(15,23,42,0.98),_rgba(2,6,23,0.96))] p-8 lg:grid-cols-[1.2fr,0.8fr]">
            <div>
              <Badge className="border-cyan-400/30 bg-cyan-400/10 text-cyan-100">FastAPI + React</Badge>
              <h2 className="mt-6 max-w-3xl text-4xl font-semibold leading-tight text-white">
                Atlas now behaves like a product, not a demo form.
              </h2>
              <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300">
                Explore cached pipeline outputs instantly, then trigger fresh YouTube analysis from the
                same workspace when you want a live run.
              </p>
              <div className="mt-8 grid gap-4 sm:grid-cols-3">
                <MetricCard label="Runs detected" value={runs.length} icon={Layers3} />
                <MetricCard label="Videos in active run" value={videos.length} icon={Workflow} />
                <MetricCard
                  label="Comparison rows"
                  value={comparison.length}
                  icon={Table2}
                />
              </div>
            </div>

            <Card className="border-white/10 bg-slate-950/50 p-6">
              <SectionHeading
                title="Run controls"
                description="Use the cached fallback immediately or launch a fresh query."
              />

              <form className="space-y-4" onSubmit={handleSubmit}>
                <label className="block">
                  <span className="mb-2 block text-sm font-medium text-slate-200">Search query</span>
                  <textarea
                    className="h-28 w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none ring-0 placeholder:text-slate-500 focus:border-cyan-300"
                    placeholder="CrewAI tutorial"
                    value={searchQuery}
                    onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
                      onSearchQueryChange(event.target.value)
                    }
                  />
                </label>

                <div className="grid gap-4 sm:grid-cols-3">
                  <label className="block">
                    <span className="mb-2 block text-sm font-medium text-slate-200">Max videos</span>
                    <input
                      className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none focus:border-cyan-300"
                      type="number"
                      min={1}
                      max={10}
                      value={maxVideos}
                      onChange={(event) => onMaxVideosChange(Number(event.target.value))}
                    />
                  </label>
                  <label className="block">
                    <span className="mb-2 block text-sm font-medium text-slate-200">
                      Transcript language
                    </span>
                    <select
                      className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none focus:border-cyan-300"
                      value={transcriptLanguage}
                      onChange={(event) => onTranscriptLanguageChange(event.target.value)}
                    >
                      {['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh'].map((language) => (
                        <option key={language} value={language} className="bg-slate-950">
                          {language}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="block">
                    <span className="mb-2 block text-sm font-medium text-slate-200">Workers</span>
                    <input
                      className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none focus:border-cyan-300"
                      type="number"
                      min={0}
                      max={16}
                      value={numWorkers}
                      onChange={(event) => onNumWorkersChange(Number(event.target.value))}
                    />
                  </label>
                </div>

                <div className="flex flex-wrap gap-3">
                  <Button className="gap-2" disabled={isSearching || !searchQuery.trim()} type="submit">
                    <PlayCircle className="size-4" />
                    {isSearching ? 'Starting run...' : 'Start fresh run'}
                  </Button>
                  <Button
                    className="gap-2"
                    disabled={!activeRunId || isGeneratingTranscripts}
                    onClick={onGenerateTranscripts}
                    variant="secondary"
                  >
                    <FileSearch className="size-4" />
                    {isGeneratingTranscripts ? 'Generating...' : 'Generate transcripts'}
                  </Button>
                  <Button
                    className="gap-2"
                    disabled={!activeRunId || isGeneratingSummaries}
                    onClick={onGenerateSummaries}
                    variant="secondary"
                  >
                    <Bot className="size-4" />
                    {isGeneratingSummaries ? 'Generating...' : 'Generate summaries'}
                  </Button>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <Button
                    className="gap-2"
                    disabled={!activeRunId || isRefreshingComparison}
                    onClick={onRefreshComparison}
                    variant="ghost"
                  >
                    <RefreshCcw className="size-4" />
                    {isRefreshingComparison ? 'Refreshing...' : 'Refresh comparison'}
                  </Button>
                  <Button
                    className="gap-2"
                    disabled={!activeRunId || isGeneratingAssignments}
                    onClick={onGenerateAssignments}
                    variant="ghost"
                  >
                    <GraduationCap className="size-4" />
                    {isGeneratingAssignments ? 'Generating...' : 'Generate assignments'}
                  </Button>
                </div>
              </form>
            </Card>
          </div>
        </Card>
      </motion.div>

      <div className="grid gap-6 xl:grid-cols-[320px,minmax(0,1fr)]">
        <Card className="p-6">
          <SectionHeading
            title="Run explorer"
            description="Switch between cached datasets and fresh executions."
          />
          <div className="space-y-3">
            {runs.map((run) => (
              <button
                key={run.run_id}
                className={cn(
                  'w-full rounded-2xl border p-4 text-left transition',
                  run.run_id === activeRunId
                    ? 'border-cyan-300/60 bg-cyan-400/10'
                    : 'border-white/10 bg-white/5 hover:bg-white/10',
                )}
                onClick={() => onSelectRun(run.run_id)}
                type="button"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-medium text-white">{run.search_query || run.run_id}</p>
                    <p className="mt-1 text-xs text-slate-400">{formatDate(run.created_at)}</p>
                  </div>
                  {run.is_fallback ? (
                    <Badge className="border-violet-400/30 bg-violet-400/10 text-violet-100">Demo</Badge>
                  ) : (
                    <Badge className="text-slate-300">Run</Badge>
                  )}
                </div>
                <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-300">
                  <span>{run.counts.videos} videos</span>
                  <span>{run.counts.summaries} summaries</span>
                  <span>{run.counts.assignments} assignments</span>
                </div>
              </button>
            ))}
          </div>
        </Card>

        <div className="space-y-6">
          {currentRun ? (
            <Card className="p-6">
              <div className="flex flex-wrap items-center gap-3">
                <Badge className="border-emerald-400/30 bg-emerald-400/10 text-emerald-100">
                  {currentRun.is_fallback ? 'Cached demo data' : 'Selected run'}
                </Badge>
                <Badge>{currentRun.search_query || currentRun.run_id}</Badge>
                <span className="text-sm text-slate-400">
                  Source: <span className="font-mono text-slate-300">{currentRun.source_folder}</span>
                </span>
              </div>
              <div className="mt-4 grid gap-4 md:grid-cols-4">
                <MetricCard label="Videos" value={currentRun.counts.videos} icon={Search} />
                <MetricCard label="Transcripts" value={currentRun.counts.transcripts} icon={FileSearch} />
                <MetricCard label="Summaries" value={currentRun.counts.summaries} icon={Sparkles} />
                <MetricCard label="Assignments" value={currentRun.counts.assignments} icon={GraduationCap} />
              </div>
            </Card>
          ) : null}

          <Card className="p-6">
            <SectionHeading
              title="Analysis workspace"
              description="Every artifact is rendered from structured FastAPI responses."
            />

            {error ? <p className="rounded-2xl border border-red-400/30 bg-red-500/10 p-4 text-sm text-red-100">{error}</p> : null}

            {isLoading ? (
              <div className="grid gap-4 md:grid-cols-2">
                {Array.from({ length: 4 }).map((_, index) => (
                  <div
                    key={index}
                    className="h-28 animate-pulse rounded-3xl border border-white/10 bg-white/5"
                  />
                ))}
              </div>
            ) : null}

            {!isLoading && bundle ? (
              <Tabs.Root className="space-y-6" defaultValue="videos">
                <Tabs.List className="flex flex-wrap gap-2 rounded-3xl border border-white/10 bg-slate-950/40 p-2">
                  {[
                    ['videos', 'Search results'],
                    ['transcripts', 'Transcripts'],
                    ['summaries', 'Summaries'],
                    ['comparison', 'Comparison'],
                    ['assignments', 'Assignments'],
                  ].map(([value, label]) => (
                    <Tabs.Trigger
                      key={value}
                      className="rounded-2xl px-4 py-2 text-sm font-medium text-slate-300 outline-none transition data-[state=active]:bg-white data-[state=active]:text-slate-950"
                      value={value}
                    >
                      {label}
                    </Tabs.Trigger>
                  ))}
                </Tabs.List>

                <Tabs.Content className="space-y-4" value="videos">
                  {videos.map((video) => (
                    <Card key={video.video_id} className="p-5">
                      <div className="flex flex-wrap items-center gap-3">
                        <Badge>{video.channel}</Badge>
                        <Badge>{video.duration}</Badge>
                        <Badge>{video.published_at.slice(0, 10)}</Badge>
                      </div>
                      <h4 className="mt-4 text-lg font-semibold text-white">{video.title}</h4>
                      <p className="mt-3 text-sm leading-7 text-slate-300">{video.description}</p>
                      <a
                        className="mt-4 inline-block text-sm font-medium text-cyan-300 hover:text-cyan-200"
                        href={video.url}
                        rel="noreferrer"
                        target="_blank"
                      >
                        Open video
                      </a>
                    </Card>
                  ))}
                </Tabs.Content>

                <Tabs.Content className="space-y-4" value="transcripts">
                  {transcripts.map((item) => (
                    <Card key={item.video_id} className="p-5">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <h4 className="text-lg font-semibold text-white">{item.title}</h4>
                          <p className="mt-1 text-sm text-slate-400">{item.channel}</p>
                        </div>
                        <Badge>{item.available ? item.language : 'Missing'}</Badge>
                      </div>
                      <p className="mt-4 max-h-80 overflow-y-auto whitespace-pre-wrap text-sm leading-7 text-slate-300">
                        {item.cleaned_text || 'No transcript available for this video yet.'}
                      </p>
                    </Card>
                  ))}
                </Tabs.Content>

                <Tabs.Content className="space-y-4" value="summaries">
                  {summaries.map((item) => (
                    <Card key={item.video_id} className="p-5">
                      <div className="flex flex-wrap items-center gap-3">
                        <h4 className="text-lg font-semibold text-white">{item.title}</h4>
                        <Badge>{item.channel}</Badge>
                      </div>
                      <p className="mt-4 text-sm leading-7 text-slate-300">{item.high_level_overview}</p>

                      <div className="mt-6 grid gap-6 xl:grid-cols-[1.2fr,0.8fr]">
                        <div>
                          <h5 className="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
                            Technical breakdown
                          </h5>
                          <ul className="space-y-3">
                            {item.technical_breakdown.map((entry, index) =>
                              renderBreakdownItem(entry, index),
                            )}
                          </ul>
                        </div>
                        <div className="space-y-4">
                          <Card className="p-4">
                            <p className="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
                              Insights
                            </p>
                            <ul className="space-y-2 text-sm text-slate-300">
                              {item.insights.map((insight) => (
                                <li key={insight}>• {insight}</li>
                              ))}
                            </ul>
                          </Card>
                          <Card className="p-4">
                            <p className="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
                              Applications
                            </p>
                            <ul className="space-y-2 text-sm text-slate-300">
                              {item.applications.map((application) => (
                                <li key={application}>• {application}</li>
                              ))}
                            </ul>
                          </Card>
                          <Card className="p-4">
                            <p className="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">
                              Limitations
                            </p>
                            <ul className="space-y-2 text-sm text-slate-300">
                              {item.limitations.map((limitation) => (
                                <li key={limitation}>• {limitation}</li>
                              ))}
                            </ul>
                          </Card>
                        </div>
                      </div>
                    </Card>
                  ))}
                </Tabs.Content>

                <Tabs.Content className="space-y-4" value="comparison">
                  <Card className="overflow-hidden">
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-white/10 text-left text-sm">
                        <thead className="bg-slate-950/60 text-slate-400">
                          <tr>
                            {[
                              'Title',
                              'Difficulty',
                              'Teaching style',
                              'Depth',
                              'Practical value',
                              'Audience',
                              'Technologies',
                            ].map((heading) => (
                              <th key={heading} className="px-4 py-4 font-medium">
                                {heading}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                          {comparison.map((row) => (
                            <tr key={row.video_id} className="align-top">
                              <td className="px-4 py-4">
                                <p className="font-medium text-white">{row.title}</p>
                                <p className="mt-2 text-xs text-slate-400">{row.channel}</p>
                              </td>
                              <td className="px-4 py-4 text-slate-200">{row.difficulty}</td>
                              <td className="px-4 py-4 text-slate-200">{row.teaching_style}</td>
                              <td className="px-4 py-4 text-slate-200">{row.content_depth}</td>
                              <td className="px-4 py-4 text-slate-200">{row.practical_value}</td>
                              <td className="px-4 py-4 text-slate-200">{row.target_audience}</td>
                              <td className="px-4 py-4 text-slate-200">
                                <div className="flex flex-wrap gap-2">
                                  {row.key_technologies.map((technology) => (
                                    <Badge key={technology} className="text-[10px]">
                                      {technology}
                                    </Badge>
                                  ))}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>

                  <div className="grid gap-4 lg:grid-cols-[0.7fr,1.3fr]">
                    <Card className="p-5">
                      <SectionHeading
                        title="Recommendations"
                        description="Safe fallback suggestions derived from cached artifacts."
                      />
                      <ul className="space-y-3 text-sm leading-7 text-slate-300">
                        {bundle.comparison.recommendations.map((recommendation) => (
                          <li key={recommendation}>• {recommendation}</li>
                        ))}
                      </ul>
                    </Card>
                    <Card className="p-5">
                      <SectionHeading
                        title="Insights report"
                        description="The backend keeps the original report-style analysis available."
                      />
                      <pre className="whitespace-pre-wrap text-sm leading-7 text-slate-300">
                        {bundle.comparison.insights_report}
                      </pre>
                    </Card>
                  </div>
                </Tabs.Content>

                <Tabs.Content className="space-y-4" value="assignments">
                  <AnimatePresence mode="popLayout">
                    {assignments.map((item) => (
                      <motion.div
                        key={item.video_id}
                        initial={{ opacity: 0, y: 18 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -18 }}
                      >
                        <Card className="p-6">
                          <div className="flex flex-wrap items-center gap-3">
                            <h4 className="text-lg font-semibold text-white">{item.title}</h4>
                            <Badge>{item.channel}</Badge>
                            <Badge className="border-emerald-400/30 bg-emerald-400/10 text-emerald-100">
                              {item.available ? 'Ready' : 'Missing'}
                            </Badge>
                          </div>
                          <div className="mt-4 flex flex-wrap gap-2 text-sm text-slate-400">
                            {Object.entries(item.metadata).map(([key, value]) => (
                              <span key={key} className="rounded-full border border-white/10 px-3 py-1">
                                {key}: {String(value)}
                              </span>
                            ))}
                          </div>
                          <div className="prose prose-invert mt-6 max-w-none prose-headings:text-white prose-p:text-slate-300 prose-strong:text-white prose-li:text-slate-300">
                            <ReactMarkdown>{item.markdown}</ReactMarkdown>
                          </div>
                        </Card>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </Tabs.Content>
              </Tabs.Root>
            ) : null}
          </Card>
        </div>
      </div>
    </section>
  )
}
