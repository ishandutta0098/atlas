import type { ButtonHTMLAttributes } from 'react'

import { cn } from '../../lib/utils'

type ButtonVariant = 'primary' | 'secondary' | 'ghost'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
}

const variants: Record<ButtonVariant, string> = {
  primary:
    'border border-white/10 bg-[#f3f4f6] text-[#101217] shadow-[0_14px_40px_rgba(255,255,255,0.12)] hover:bg-white',
  secondary:
    'border border-white/10 bg-white/[0.04] text-zinc-100 hover:bg-white/[0.07]',
  ghost: 'bg-transparent text-zinc-300 hover:bg-white/[0.05]',
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
        'inline-flex items-center justify-center rounded-2xl px-4 py-2.5 text-sm font-medium transition duration-200 disabled:cursor-not-allowed disabled:opacity-60',
        variants[variant],
        className,
      )}
      type={type}
      {...props}
    />
  )
}
