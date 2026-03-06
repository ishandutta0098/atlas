import type { ButtonHTMLAttributes } from 'react'

import { cn } from '../../lib/utils'

type ButtonVariant = 'primary' | 'secondary' | 'ghost'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
}

const variants: Record<ButtonVariant, string> = {
  primary:
    'bg-gradient-to-r from-cyan-400 via-blue-500 to-violet-500 text-white shadow-[0_18px_40px_rgba(56,189,248,0.28)] hover:brightness-110',
  secondary:
    'border border-white/10 bg-white/10 text-slate-100 hover:bg-white/15',
  ghost: 'bg-transparent text-slate-300 hover:bg-white/10',
}

export function Button({
  className,
  variant = 'primary',
  type = 'button',
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center rounded-2xl px-4 py-2.5 text-sm font-semibold transition duration-200 disabled:cursor-not-allowed disabled:opacity-60',
        variants[variant],
        className,
      )}
      type={type}
      {...props}
    />
  )
}
