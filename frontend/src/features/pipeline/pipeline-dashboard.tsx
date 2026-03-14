import * as Tabs from '@radix-ui/react-tabs'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowUpRight,
  CheckCheck,
  Circle,
  Layers3,
  LoaderCircle,
  Workflow,
} from 'lucide-react'
import {
  type ChangeEvent,
  type FormEvent,
  type ReactNode,
  useEffect,
  useState,
} from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { Badge } from '../../components/ui/badge'
import { Button } from '../../components/ui/button'
import { Card } from '../../components/ui/card'
import type { AssignmentArtifact, RunBundle } from '../../lib/types'
import { cn } from '../../lib/utils'

interface PipelineDashboardProps {
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
  onStartRun: () => void
  isRunning: boolean
}

type ProgressMap = Record<string, boolean>

function SectionHeading({
  title,
  description,
  action,
}: {
  title: string
  description?: string
  action?: ReactNode
}) {
  return (
    <div className="mb-5 flex flex-col gap-2 lg:flex-row lg:items-end lg:justify-between">
      <div>
        <h3 className="text-lg font-semibold tracking-[-0.03em] text-white">{title}</h3>
        {description ? <p className="mt-1 text-sm text-zinc-400">{description}</p> : null}
      </div>
      {action}
    </div>
  )
}

function MarkdownBody({ markdown }: { markdown: string }) {
  return (
    <div className="space-y-4 text-sm leading-7 text-zinc-300">
      <ReactMarkdown
        components={{
          a: ({ className, ...props }) => (
            <a
              className={cn('font-medium text-white underline decoration-white/20 underline-offset-4', className)}
              rel="noreferrer"
              target="_blank"
              {...props}
            />
          ),
          h1: ({ className, ...props }) => (
            <h1 className={cn('text-2xl font-semibold tracking-[-0.04em] text-white', className)} {...props} />
          ),
          h2: ({ className, ...props }) => (
            <h2 className={cn('text-xl font-semibold tracking-[-0.03em] text-white', className)} {...props} />
          ),
          h3: ({ className, ...props }) => (
            <h3 className={cn('text-base font-semibold text-white', className)} {...props} />
          ),
          p: ({ className, ...props }) => <p className={cn('text-zinc-300', className)} {...props} />,
          ul: ({ className, ...props }) => <ul className={cn('space-y-2 pl-5', className)} {...props} />,
          ol: ({ className, ...props }) => <ol className={cn('space-y-2 pl-5', className)} {...props} />,
          li: ({ className, ...props }) => <li className={cn('text-zinc-300', className)} {...props} />,
          pre: ({ className, ...props }) => (
            <pre
              className={cn(
                'overflow-x-auto rounded-2xl border border-white/8 bg-black/20 p-4 text-[13px] leading-6 text-zinc-100',
                className,
              )}
              {...props}
            />
          ),
          code: ({ className, ...props }) => (
            <code
              className={cn('rounded-md bg-white/[0.05] px-1.5 py-0.5 text-[0.92em] text-zinc-100', className)}
              {...props}
            />
          ),
        }}
        remarkPlugins={[remarkGfm]}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  )
}

function formatDate(value: string) {
  if (!value) return 'Unavailable'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

function getVideoThumbnail(videoId: string) {
  return `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`
}

function trimText(value: string, maxLength: number) {
  if (value.length <= maxLength) return value
  return `${value.slice(0, maxLength - 1).trimEnd()}...`
}

function splitBreakdown(entries: Array<Record<string, unknown>>) {
  const tools = entries.filter((entry) => entry.type === 'tool')
  const processes = entries.filter((entry) => entry.type === 'process')
  const architecture = entries.filter((entry) => entry.type !== 'tool' && entry.type !== 'process')
  return { tools, processes, architecture }
}

function getAssignmentStorageKey(runId: string, videoId: string) {
  return `atlas-assignment-progress:${runId}:${videoId}`
}

function getAssignmentProgressItems(item: AssignmentArtifact) {
  if (item.checklist.length > 0) {
    return item.checklist
  }

  if (item.sections.length > 0) {
    return item.sections.map((section) => ({
      id: section.id,
      label: section.title,
    }))
  }

  return item.markdown
    ? [
        {
          id: 'review-assignment',
          label: 'Review assignment',
        },
      ]
    : []
}

export function PipelineDashboard({
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
  onStartRun,
  isRunning,
}: PipelineDashboardProps) {
  const videos = bundle?.videos.videos ?? []
  const transcripts = bundle?.transcripts.items ?? []
  const summaries = bundle?.summaries.items ?? []
  const comparison = bundle?.comparison.rows ?? []
  const assignments = bundle?.assignments.items ?? []
  const [assignmentProgress, setAssignmentProgress] = useState<Record<string, ProgressMap>>({})

  useEffect(() => {
    if (!activeRunId) {
      setAssignmentProgress({})
      return
    }

    const nextState: Record<string, ProgressMap> = {}
    for (const item of assignments) {
      const saved = localStorage.getItem(getAssignmentStorageKey(activeRunId, item.video_id))
      if (!saved) {
        nextState[item.video_id] = {}
        continue
      }

      try {
        nextState[item.video_id] = JSON.parse(saved) as ProgressMap
      } catch {
        nextState[item.video_id] = {}
      }
    }

    setAssignmentProgress(nextState)
  }, [activeRunId, assignments])

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    onStartRun()
  }

  const toggleAssignmentItem = (videoId: string, itemId: string) => {
    if (!activeRunId) return

    setAssignmentProgress((current) => {
      const nextVideoState = {
        ...(current[videoId] ?? {}),
        [itemId]: !(current[videoId] ?? {})[itemId],
      }
      localStorage.setItem(
        getAssignmentStorageKey(activeRunId, videoId),
        JSON.stringify(nextVideoState),
      )
      return {
        ...current,
        [videoId]: nextVideoState,
      }
    })
  }

  return (
    <section id="pipeline" className="space-y-8">
      <motion.div
        animate={{ opacity: 1, y: 0 }}
        initial={{ opacity: 0, y: 20 }}
        transition={{ duration: 0.4 }}
      >
        <Card className="mx-auto max-w-6xl overflow-hidden border-white/8 bg-[linear-gradient(180deg,rgba(24,27,33,0.96),rgba(16,18,22,0.92))] p-6 md:p-8">
          <form onSubmit={handleSubmit}>
            <div className="flex flex-col gap-3 xl:flex-row xl:items-end">
              <label className="block xl:flex-[1.8]">
                <span className="mb-2 block text-xs uppercase tracking-[0.2em] text-zinc-500">
                  Search query
                </span>
                <input
                  className="h-12 w-full rounded-2xl border border-white/8 bg-white/[0.04] px-4 text-sm text-white outline-none placeholder:text-zinc-500 focus:border-white/15"
                  placeholder="CrewAI tutorial"
                  value={searchQuery}
                  onChange={(event: ChangeEvent<HTMLInputElement>) =>
                    onSearchQueryChange(event.target.value)
                  }
                />
              </label>

              <label className="block xl:w-36">
                <span className="mb-2 block text-xs uppercase tracking-[0.2em] text-zinc-500">
                  Max videos
                </span>
                <input
                  className="h-12 w-full rounded-2xl border border-white/8 bg-white/[0.04] px-4 text-sm text-white outline-none focus:border-white/15"
                  max={10}
                  min={1}
                  type="number"
                  value={maxVideos}
                  onChange={(event) => onMaxVideosChange(Number(event.target.value))}
                />
              </label>

              <label className="block xl:w-44">
                <span className="mb-2 block text-xs uppercase tracking-[0.2em] text-zinc-500">
                  Transcript language
                </span>
                <select
                  className="h-12 w-full rounded-2xl border border-white/8 bg-white/[0.04] px-4 text-sm text-white outline-none focus:border-white/15"
                  value={transcriptLanguage}
                  onChange={(event) => onTranscriptLanguageChange(event.target.value)}
                >
                  {['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh'].map((language) => (
                    <option key={language} className="bg-[#13161a]" value={language}>
                      {language}
                    </option>
                  ))}
                </select>
              </label>

              <label className="block xl:w-32">
                <span className="mb-2 block text-xs uppercase tracking-[0.2em] text-zinc-500">
                  Workers
                </span>
                <input
                  className="h-12 w-full rounded-2xl border border-white/8 bg-white/[0.04] px-4 text-sm text-white outline-none focus:border-white/15"
                  max={16}
                  min={0}
                  type="number"
                  value={numWorkers}
                  onChange={(event) => onNumWorkersChange(Number(event.target.value))}
                />
              </label>

              <Button
                className="h-12 gap-2 px-5 xl:min-w-[170px]"
                disabled={isRunning || !searchQuery.trim()}
                type="submit"
              >
                {isRunning ? <LoaderCircle className="size-4 animate-spin" /> : <Workflow className="size-4" />}
                {isRunning ? 'Running...' : 'Run'}
              </Button>
            </div>
          </form>
        </Card>
      </motion.div>

      <Card className="p-6">
        <SectionHeading
          title="Analysis workspace"
          description="Structured video search, transcripts, summaries, comparison, and assignments."
        />

        {error ? (
          <p className="rounded-2xl border border-red-400/20 bg-red-500/8 p-4 text-sm text-red-100">
            {error}
          </p>
        ) : null}

        {isLoading ? (
          <div className="grid gap-4 md:grid-cols-2">
            {Array.from({ length: 4 }).map((_, index) => (
              <div
                key={index}
                className="h-28 animate-pulse rounded-[24px] border border-white/8 bg-white/[0.04]"
              />
            ))}
          </div>
        ) : null}

        {!isLoading && bundle ? (
          <Tabs.Root className="space-y-6" defaultValue="videos">
            <Tabs.List className="flex flex-wrap gap-2 rounded-3xl border border-white/8 bg-black/10 p-2">
              {[
                ['videos', 'Search results'],
                ['transcripts', 'Transcripts'],
                ['summaries', 'Summaries'],
                ['comparison', 'Comparison'],
                ['assignments', 'Assignments'],
              ].map(([value, label]) => (
                <Tabs.Trigger
                  key={value}
                  className="rounded-2xl px-4 py-2 text-sm font-medium text-zinc-400 outline-none transition data-[state=active]:bg-white data-[state=active]:text-[#101217]"
                  value={value}
                >
                  {label}
                </Tabs.Trigger>
              ))}
            </Tabs.List>

            <Tabs.Content className="space-y-4" value="videos">
              <div className="grid gap-5 xl:grid-cols-2">
                {videos.map((video) => (
                  <Card key={video.video_id} className="overflow-hidden p-0">
                    <div className="grid gap-0 md:grid-cols-[280px,minmax(0,1fr)]">
                      <div className="relative aspect-video bg-black/30 md:aspect-auto">
                        <img
                          alt={video.title}
                          className="h-full w-full object-cover"
                          src={getVideoThumbnail(video.video_id)}
                        />
                        <div className="absolute bottom-3 right-3 rounded-lg bg-black/70 px-2 py-1 text-xs text-white">
                          {video.duration}
                        </div>
                      </div>
                      <div className="p-5">
                        <div className="flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-zinc-500">
                          <span>{video.channel}</span>
                          <span className="text-zinc-700">/</span>
                          <span>{formatDate(video.published_at)}</span>
                        </div>
                        <h4 className="mt-3 text-xl font-semibold tracking-[-0.03em] text-white">
                          {video.title}
                        </h4>
                        <p className="mt-3 text-sm leading-7 text-zinc-300">
                          {trimText(video.description || 'No description available for this result.', 240)}
                        </p>
                        <div className="mt-5 flex items-center justify-between gap-3">
                          <div className="flex flex-wrap gap-2">
                            <Badge>{video.channel}</Badge>
                            <Badge>{video.duration}</Badge>
                          </div>
                          <a
                            className="inline-flex items-center gap-2 text-sm font-medium text-white"
                            href={video.url}
                            rel="noreferrer"
                            target="_blank"
                          >
                            Watch
                            <ArrowUpRight className="size-4" />
                          </a>
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </Tabs.Content>

            <Tabs.Content className="space-y-4" value="transcripts">
              {transcripts.map((item) => (
                <Card key={item.video_id} className="p-5">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <h4 className="text-lg font-semibold text-white">{item.title}</h4>
                      <p className="mt-1 text-sm text-zinc-400">{item.channel}</p>
                    </div>
                    <Badge>{item.available ? item.language : 'Missing'}</Badge>
                  </div>
                  <p className="mt-4 max-h-[420px] overflow-y-auto whitespace-pre-wrap text-sm leading-7 text-zinc-300">
                    {item.cleaned_text || 'No transcript available for this video yet.'}
                  </p>
                </Card>
              ))}
            </Tabs.Content>

            <Tabs.Content className="space-y-5" value="summaries">
              {summaries.map((item) => {
                const breakdown = splitBreakdown(item.technical_breakdown)

                return (
                  <Card key={item.video_id} className="overflow-hidden p-0">
                    <div className="border-b border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.01))] px-6 py-5">
                      <div className="flex flex-wrap items-center gap-3">
                        <h4 className="text-xl font-semibold tracking-[-0.03em] text-white">{item.title}</h4>
                        <Badge>{item.channel}</Badge>
                      </div>
                    </div>

                    <div className="space-y-6 p-6">
                      <div className="grid gap-6 xl:grid-cols-[1.2fr,0.8fr]">
                        <div className="rounded-[28px] border border-white/8 bg-[linear-gradient(145deg,rgba(255,255,255,0.05),rgba(255,255,255,0.015))] p-6">
                          <p className="text-[11px] uppercase tracking-[0.24em] text-zinc-500">Overview</p>
                          <p className="mt-5 max-w-3xl text-[15px] leading-8 text-zinc-200">
                            {item.high_level_overview}
                          </p>
                        </div>

                        <div className="rounded-[28px] border border-white/8 bg-black/15 p-6">
                          <SectionHeading
                            title="Architecture"
                            description="System structure and orchestration patterns."
                          />
                          {breakdown.architecture.length > 0 ? (
                            <div className="space-y-4">
                              {breakdown.architecture.map((entry, index) => (
                                <div
                                  key={`${item.video_id}-architecture-${index}`}
                                  className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5"
                                >
                                  <p className="text-sm leading-7 text-zinc-300">
                                    {String(entry.description ?? 'Architecture detail')}
                                  </p>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <p className="text-sm text-zinc-400">No explicit architecture notes were extracted.</p>
                          )}
                        </div>
                      </div>

                      <div className="grid gap-6 xl:grid-cols-[0.95fr,1.05fr]">
                        <div className="rounded-[28px] border border-white/8 bg-black/15 p-6">
                          <SectionHeading
                            title="Process flow"
                            description="The main implementation sequence broken into steps."
                          />
                          <div className="space-y-4">
                            {breakdown.processes.map((entry, index) => (
                              <div
                                key={`${item.video_id}-process-${index}`}
                                className="relative overflow-hidden rounded-[24px] border border-white/8 bg-white/[0.03] p-5"
                              >
                                <div className="absolute inset-y-0 left-0 w-px bg-white/10" />
                                <div className="pl-4">
                                  <p className="text-[11px] uppercase tracking-[0.22em] text-zinc-500">
                                    Step {String(entry.step_number ?? index + 1)}
                                  </p>
                                  <p className="mt-3 text-sm leading-7 text-zinc-300">
                                    {String(entry.description ?? '')}
                                  </p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="rounded-[28px] border border-white/8 bg-black/15 p-6">
                          <SectionHeading
                            title="Tools"
                            description="Frameworks, models, and systems referenced in the walkthrough."
                          />
                          <div className="grid gap-4 md:grid-cols-2">
                            {breakdown.tools.map((entry, index) => (
                              <div
                                key={`${item.video_id}-tool-${index}`}
                                className="rounded-[24px] border border-white/8 bg-white/[0.03] p-5"
                              >
                                <p className="text-sm font-medium text-white">
                                  {String(entry.name ?? 'Tool')}
                                </p>
                                <p className="mt-3 text-sm leading-7 text-zinc-300">
                                  {String(entry.purpose ?? '')}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>

                      <div className="grid gap-6 xl:grid-cols-3">
                        <div className="rounded-[28px] border border-emerald-400/12 bg-emerald-400/[0.04] p-6">
                          <SectionHeading title="Insights" description="Key takeaways and notable design decisions." />
                          <ul className="space-y-3">
                            {item.insights.map((insight, index) => (
                              <li
                                key={insight}
                                className="rounded-[20px] border border-white/8 bg-black/10 px-4 py-4 text-sm leading-7 text-zinc-300"
                              >
                                <span className="mr-3 text-zinc-500">{String(index + 1).padStart(2, '0')}</span>
                                {insight}
                              </li>
                            ))}
                          </ul>
                        </div>

                        <div className="rounded-[28px] border border-sky-400/12 bg-sky-400/[0.04] p-6">
                          <SectionHeading title="Applications" description="How the concepts translate into practical use." />
                          <ul className="space-y-3">
                            {item.applications.map((application, index) => (
                              <li
                                key={application}
                                className="rounded-[20px] border border-white/8 bg-black/10 px-4 py-4 text-sm leading-7 text-zinc-300"
                              >
                                <span className="mr-3 text-zinc-500">{String(index + 1).padStart(2, '0')}</span>
                                {application}
                              </li>
                            ))}
                          </ul>
                        </div>

                        <div className="rounded-[28px] border border-amber-400/12 bg-amber-400/[0.04] p-6">
                          <SectionHeading title="Limitations" description="Trade-offs, caveats, and implementation constraints." />
                          <ul className="space-y-3">
                            {item.limitations.map((limitation, index) => (
                              <li
                                key={limitation}
                                className="rounded-[20px] border border-white/8 bg-black/10 px-4 py-4 text-sm leading-7 text-zinc-300"
                              >
                                <span className="mr-3 text-zinc-500">{String(index + 1).padStart(2, '0')}</span>
                                {limitation}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  </Card>
                )
              })}
            </Tabs.Content>

            <Tabs.Content className="space-y-4" value="comparison">
              <Card className="overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-white/8 text-left text-sm">
                    <thead className="bg-black/10 text-zinc-500">
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
                    <tbody className="divide-y divide-white/6">
                      {comparison.map((row) => (
                        <tr key={row.video_id} className="align-top">
                          <td className="px-4 py-4">
                            <p className="font-medium text-white">{row.title}</p>
                            <p className="mt-2 text-xs text-zinc-500">{row.channel}</p>
                          </td>
                          <td className="px-4 py-4 text-zinc-200">{row.difficulty}</td>
                          <td className="px-4 py-4 text-zinc-200">{row.teaching_style}</td>
                          <td className="px-4 py-4 text-zinc-200">{row.content_depth}</td>
                          <td className="px-4 py-4 text-zinc-200">{row.practical_value}</td>
                          <td className="px-4 py-4 text-zinc-200">{row.target_audience}</td>
                          <td className="px-4 py-4 text-zinc-200">
                            <div className="flex flex-wrap gap-2">
                              {row.key_technologies.map((technology) => (
                                <Badge key={technology}>{technology}</Badge>
                              ))}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              <div className="grid gap-4 lg:grid-cols-[0.72fr,1.28fr]">
                <Card className="p-5">
                  <SectionHeading title="Recommendations" />
                  <ul className="space-y-3 text-sm leading-7 text-zinc-300">
                    {bundle.comparison.recommendations.map((recommendation) => (
                      <li key={recommendation}>• {recommendation}</li>
                    ))}
                  </ul>
                </Card>
                <Card className="p-5">
                  <SectionHeading title="Insights report" />
                  <pre className="whitespace-pre-wrap text-sm leading-7 text-zinc-300">
                    {bundle.comparison.insights_report}
                  </pre>
                </Card>
              </div>
            </Tabs.Content>

            <Tabs.Content className="space-y-5" value="assignments">
              <AnimatePresence mode="popLayout">
                {assignments.map((item) => {
                  const progressItems = getAssignmentProgressItems(item)
                  const completedCount = progressItems.filter(
                    (progressItem) => assignmentProgress[item.video_id]?.[progressItem.id],
                  ).length
                  const progressPercent =
                    progressItems.length > 0
                      ? Math.round((completedCount / progressItems.length) * 100)
                      : 0

                  return (
                    <motion.div
                      key={item.video_id}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -18 }}
                      initial={{ opacity: 0, y: 18 }}
                    >
                      <Card className="p-6">
                        <div className="flex flex-wrap items-center gap-3">
                          <h4 className="text-xl font-semibold tracking-[-0.03em] text-white">
                            {item.title}
                          </h4>
                          <Badge>{item.channel}</Badge>
                          <Badge>{item.available ? 'Ready' : 'Missing'}</Badge>
                        </div>

                        <div className="mt-4 flex flex-wrap gap-2">
                          {Object.entries(item.display_metadata).map(([key, value]) => (
                            <Badge key={key}>
                              {key.replaceAll('_', ' ')}: {value}
                            </Badge>
                          ))}
                        </div>

                        <div className="mt-6 grid gap-6 xl:grid-cols-[340px,minmax(0,1fr)]">
                          <Card className="h-fit p-5">
                            <div className="flex items-center justify-between gap-4">
                              <div>
                                <p className="text-[11px] uppercase tracking-[0.22em] text-zinc-500">
                                  Progress
                                </p>
                                <p className="mt-2 text-3xl font-semibold tracking-[-0.04em] text-white">
                                  {progressPercent}%
                                </p>
                              </div>
                              <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-3 text-white">
                                <CheckCheck className="size-5" />
                              </div>
                            </div>

                            <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/[0.06]">
                              <div
                                className="h-full rounded-full bg-white transition-[width]"
                                style={{ width: `${progressPercent}%` }}
                              />
                            </div>

                            <div className="mt-6 space-y-3">
                              {progressItems.map((progressItem) => {
                                const checked = Boolean(
                                  assignmentProgress[item.video_id]?.[progressItem.id],
                                )

                                return (
                                  <button
                                    key={progressItem.id}
                                    className="flex w-full items-start gap-3 rounded-2xl border border-white/8 bg-black/10 px-3 py-3 text-left transition hover:bg-white/[0.03]"
                                    onClick={() => toggleAssignmentItem(item.video_id, progressItem.id)}
                                    type="button"
                                  >
                                    <span className="pt-0.5 text-white">
                                      {checked ? (
                                        <CheckCheck className="size-4" />
                                      ) : (
                                        <Circle className="size-4" />
                                      )}
                                    </span>
                                    <span className={cn('text-sm leading-6 text-zinc-300', checked && 'text-white')}>
                                      {progressItem.label}
                                    </span>
                                  </button>
                                )
                              })}
                            </div>
                          </Card>

                          <div className="space-y-4">
                            {item.sections.length > 0 ? (
                              item.sections.map((section) => (
                                <Card key={section.id} className="p-5">
                                  <SectionHeading title={section.title} />
                                  <MarkdownBody markdown={section.markdown} />
                                </Card>
                              ))
                            ) : (
                              <Card className="p-5">
                                <SectionHeading title="Assignment" />
                                <MarkdownBody markdown={item.markdown || 'No assignment content available.'} />
                              </Card>
                            )}
                          </div>
                        </div>
                      </Card>
                    </motion.div>
                  )
                })}
              </AnimatePresence>
            </Tabs.Content>
          </Tabs.Root>
        ) : null}

        {!isLoading && !bundle ? (
          <Card className="border-dashed border-white/10 p-10 text-center">
            <div className="mx-auto flex max-w-md flex-col items-center">
              <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-3 text-white">
                <Layers3 className="size-5" />
              </div>
              <h3 className="mt-4 text-lg font-semibold text-white">No workspace loaded</h3>
              <p className="mt-2 text-sm leading-7 text-zinc-400">
                Start a run to populate the workspace with YouTube search results, summaries, comparison,
                and assignments.
              </p>
            </div>
          </Card>
        ) : null}
      </Card>
    </section>
  )
}
