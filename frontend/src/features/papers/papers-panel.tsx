import { BookOpenText, FileText, Search } from 'lucide-react'
import { type FormEvent } from 'react'

import { Badge } from '../../components/ui/badge'
import { Button } from '../../components/ui/button'
import { Card } from '../../components/ui/card'
import type { PapersQueryResponse, PapersStatusResponse } from '../../lib/types'

interface PapersPanelProps {
  status?: PapersStatusResponse
  result?: PapersQueryResponse
  query: string
  onQueryChange: (value: string) => void
  onSubmit: () => void
  isLoading: boolean
}

export function PapersPanel({
  status,
  result,
  query,
  onQueryChange,
  onSubmit,
  isLoading,
}: PapersPanelProps) {
  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    onSubmit()
  }

  return (
    <section id="papers" className="space-y-6">
      <Card className="overflow-hidden">
        <div className="grid gap-8 bg-[radial-gradient(circle_at_top_left,_rgba(34,197,94,0.16),_transparent_32%),radial-gradient(circle_at_bottom_right,_rgba(59,130,246,0.18),_transparent_34%),linear-gradient(180deg,_rgba(15,23,42,0.98),_rgba(2,6,23,0.96))] p-8 lg:grid-cols-[0.85fr,1.15fr]">
          <div>
            <Badge className="border-emerald-400/30 bg-emerald-400/10 text-emerald-100">
              Academic papers RAG
            </Badge>
            <h2 className="mt-5 text-3xl font-semibold text-white">
              Search indexed papers with structured citations.
            </h2>
            <p className="mt-4 max-w-xl text-sm leading-7 text-slate-300">
              The FastAPI layer keeps the paper query experience separate from the YouTube pipeline and
              returns source metadata, excerpts, and latency as typed data for the UI.
            </p>
            <div className="mt-6 grid gap-3 sm:grid-cols-3">
              <Card className="p-4">
                <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Ready</p>
                <p className="mt-2 text-xl font-semibold text-white">{status?.ready ? 'Yes' : 'No'}</p>
              </Card>
              <Card className="p-4">
                <p className="text-xs uppercase tracking-[0.24em] text-slate-400">PDFs</p>
                <p className="mt-2 text-xl font-semibold text-white">{status?.pdf_count ?? 0}</p>
              </Card>
              <Card className="p-4">
                <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Index</p>
                <p className="mt-2 text-xl font-semibold text-white">
                  {status?.index_exists ? 'Available' : 'Missing'}
                </p>
              </Card>
            </div>
          </div>

          <Card className="border-white/10 bg-slate-950/50 p-6">
            <form className="space-y-4" onSubmit={handleSubmit}>
              <div>
                <label className="mb-2 block text-sm font-medium text-slate-200">Ask the paper index</label>
                <textarea
                  className="h-32 w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white outline-none placeholder:text-slate-500 focus:border-emerald-300"
                  placeholder="What are the main architectures for AI agents?"
                  value={query}
                  onChange={(event) => onQueryChange(event.target.value)}
                />
              </div>
              <Button className="gap-2" disabled={isLoading || !query.trim()} type="submit">
                <Search className="size-4" />
                {isLoading ? 'Searching papers...' : 'Search papers'}
              </Button>
              {status?.message ? <p className="text-sm text-amber-200">{status.message}</p> : null}
            </form>
          </Card>
        </div>
      </Card>

      {result ? (
        <div className="grid gap-6 xl:grid-cols-[0.85fr,1.15fr]">
          <Card className="p-6">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-3 text-emerald-100">
                <BookOpenText className="size-5" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">Answer</h3>
                <p className="text-sm text-slate-400">
                  {result.num_sources} sources in {result.search_time.toFixed(2)}s
                </p>
              </div>
            </div>
            <p className="mt-5 whitespace-pre-wrap text-sm leading-7 text-slate-300">{result.response}</p>
          </Card>

          <Card className="p-6">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/10 p-3 text-cyan-100">
                <FileText className="size-5" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">Citations</h3>
                <p className="text-sm text-slate-400">Relevant excerpts from the indexed papers.</p>
              </div>
            </div>
            <div className="mt-5 space-y-4">
              {result.sources.map((source, index) => (
                <Card key={`${source.file_name}-${index}`} className="p-4">
                  <div className="flex flex-wrap gap-2">
                    <Badge>{source.title || source.file_name}</Badge>
                    <Badge>{source.authors || 'Unknown authors'}</Badge>
                    <Badge>Score {source.score.toFixed(3)}</Badge>
                  </div>
                  <p className="mt-3 text-sm leading-7 text-slate-300">{source.text}</p>
                </Card>
              ))}
            </div>
          </Card>
        </div>
      ) : null}
    </section>
  )
}
