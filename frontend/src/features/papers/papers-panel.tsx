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
        <div className="grid gap-8 bg-[linear-gradient(180deg,rgba(24,27,33,0.96),rgba(16,18,22,0.92))] p-8 lg:grid-cols-[0.85fr,1.15fr]">
          <div>
            <Badge>Academic papers RAG</Badge>
            <h2 className="mt-5 text-3xl font-semibold tracking-[-0.04em] text-white">
              Search indexed papers with structured citations.
            </h2>
            <p className="mt-4 max-w-xl text-sm leading-7 text-zinc-300">
              Query the indexed paper collection and inspect the supporting excerpts without leaving the
              Atlas workspace.
            </p>
            <div className="mt-6 grid gap-3 sm:grid-cols-3">
              <Card className="p-4">
                <p className="text-xs uppercase tracking-[0.24em] text-zinc-500">Ready</p>
                <p className="mt-2 text-xl font-semibold text-white">{status?.ready ? 'Yes' : 'No'}</p>
              </Card>
              <Card className="p-4">
                <p className="text-xs uppercase tracking-[0.24em] text-zinc-500">PDFs</p>
                <p className="mt-2 text-xl font-semibold text-white">{status?.pdf_count ?? 0}</p>
              </Card>
              <Card className="p-4">
                <p className="text-xs uppercase tracking-[0.24em] text-zinc-500">Index</p>
                <p className="mt-2 text-xl font-semibold text-white">
                  {status?.index_exists ? 'Available' : 'Missing'}
                </p>
              </Card>
            </div>
          </div>

          <Card className="border-white/8 bg-black/10 p-6">
            <form className="space-y-4" onSubmit={handleSubmit}>
              <div>
                <label className="mb-2 block text-sm font-medium text-zinc-200">Ask the paper index</label>
                <textarea
                  className="h-32 w-full rounded-2xl border border-white/8 bg-white/[0.04] px-4 py-3 text-sm text-white outline-none placeholder:text-zinc-500 focus:border-white/15"
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
              <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-3 text-white">
                <BookOpenText className="size-5" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">Answer</h3>
                <p className="text-sm text-zinc-400">
                  {result.num_sources} sources in {result.search_time.toFixed(2)}s
                </p>
              </div>
            </div>
            <p className="mt-5 whitespace-pre-wrap text-sm leading-7 text-zinc-300">{result.response}</p>
          </Card>

          <Card className="p-6">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl border border-white/8 bg-white/[0.04] p-3 text-white">
                <FileText className="size-5" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">Citations</h3>
                <p className="text-sm text-zinc-400">Relevant excerpts from the indexed papers.</p>
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
                  <p className="mt-3 text-sm leading-7 text-zinc-300">{source.text}</p>
                </Card>
              ))}
            </div>
          </Card>
        </div>
      ) : null}
    </section>
  )
}
