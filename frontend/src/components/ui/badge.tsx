import type { HTMLAttributes } from 'react'

import { cn } from '../../lib/utils'

export function Badge({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full border border-white/10 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-slate-200',
        className,
      )}
      {...props}
    />
  )
}
