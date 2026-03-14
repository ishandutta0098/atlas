import type { HTMLAttributes } from 'react'

import { cn } from '../../lib/utils'

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        'rounded-[28px] border border-white/7 bg-[#14171c]/88 shadow-[0_22px_70px_rgba(0,0,0,0.32)] backdrop-blur-xl',
        className,
      )}
      {...props}
    />
  )
}
