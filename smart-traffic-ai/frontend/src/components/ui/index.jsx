// frontend/src/components/ui/index.jsx
/**
 * Shared reusable UI primitives.
 * Import like: import { Badge, Spinner, EmptyState } from '@/components/ui'
 */

import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, AlertCircle, CheckCircle, Info, AlertTriangle } from 'lucide-react'

// ── Spinner ────────────────────────────────────────────────────
export function Spinner({ size = 'md', className = '' }) {
  const sizes = { sm: 'w-4 h-4', md: 'w-6 h-6', lg: 'w-10 h-10' }
  return (
    <span className={`inline-block border-2 border-white/20 border-t-crimson-500 rounded-full animate-spin ${sizes[size]} ${className}`} />
  )
}

// ── PageLoader ─────────────────────────────────────────────────
export function PageLoader() {
  return (
    <div className="flex items-center justify-center h-full min-h-[300px]">
      <Spinner size="lg" />
    </div>
  )
}

// ── Badge ──────────────────────────────────────────────────────
export function Badge({ children, variant = 'default', pulse = false }) {
  const variants = {
    default:   'bg-white/10 text-slate-300 border-white/15',
    low:       'bg-green-500/15 text-green-400 border-green-500/25',
    medium:    'bg-amber-500/15 text-amber-400 border-amber-500/25',
    high:      'bg-red-500/15 text-red-400 border-red-500/25',
    emergency: 'bg-red-600/20 text-red-400 border-red-600/40',
    success:   'bg-green-500/15 text-green-400 border-green-500/25',
    warning:   'bg-amber-500/15 text-amber-400 border-amber-500/25',
    error:     'bg-red-500/15 text-red-400 border-red-500/25',
    info:      'bg-blue-500/15 text-blue-400 border-blue-500/25',
  }
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-mono px-2.5 py-0.5 rounded-full border ${variants[variant] || variants.default} ${pulse ? 'animate-pulse' : ''}`}>
      {children}
    </span>
  )
}

// ── EmptyState ─────────────────────────────────────────────────
export function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center px-4">
      {Icon && (
        <div className="w-14 h-14 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-4">
          <Icon className="w-6 h-6 text-slate-600" />
        </div>
      )}
      <p className="font-display font-semibold text-slate-300 mb-1">{title}</p>
      {description && <p className="text-sm text-slate-500 max-w-xs">{description}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}

// ── Alert ──────────────────────────────────────────────────────
const ALERT_STYLES = {
  info:    { bg: 'bg-blue-500/8 border-blue-500/20',    icon: Info,          text: 'text-blue-300' },
  success: { bg: 'bg-green-500/8 border-green-500/20',  icon: CheckCircle,   text: 'text-green-300' },
  warning: { bg: 'bg-amber-500/8 border-amber-500/20',  icon: AlertTriangle, text: 'text-amber-300' },
  error:   { bg: 'bg-red-500/8 border-red-500/20',      icon: AlertCircle,   text: 'text-red-300' },
}

export function Alert({ type = 'info', title, children, onClose }) {
  const s = ALERT_STYLES[type] || ALERT_STYLES.info
  const Icon = s.icon
  return (
    <div className={`flex gap-3 p-4 rounded-xl border ${s.bg}`}>
      <Icon className={`w-4 h-4 flex-shrink-0 mt-0.5 ${s.text}`} />
      <div className="flex-1 min-w-0">
        {title && <p className={`font-medium text-sm ${s.text} mb-0.5`}>{title}</p>}
        <div className={`text-sm ${s.text} opacity-80`}>{children}</div>
      </div>
      {onClose && (
        <button onClick={onClose} className={`${s.text} opacity-60 hover:opacity-100 transition-opacity flex-shrink-0`}>
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  )
}

// ── Modal ──────────────────────────────────────────────────────
export function Modal({ open, onClose, title, children, maxWidth = 'max-w-lg' }) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
          onClick={e => e.target === e.currentTarget && onClose?.()}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 16 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 8 }}
            transition={{ duration: 0.2 }}
            className={`glass-card w-full ${maxWidth} overflow-hidden`}
          >
            {title && (
              <div className="flex items-center justify-between px-6 py-4 border-b border-white/8">
                <h3 className="font-display font-semibold text-base text-white">{title}</h3>
                {onClose && (
                  <button onClick={onClose} className="p-1.5 text-slate-500 hover:text-white rounded-lg hover:bg-white/8 transition-colors">
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>
            )}
            <div className="p-6">{children}</div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

// ── ProgressBar ────────────────────────────────────────────────
export function ProgressBar({ value = 0, max = 100, color = '#EF4444', height = 'h-2', label, showValue = false }) {
  const pct = Math.min(Math.max((value / max) * 100, 0), 100)
  return (
    <div>
      {(label || showValue) && (
        <div className="flex justify-between mb-1.5">
          {label && <span className="text-xs text-slate-400">{label}</span>}
          {showValue && <span className="text-xs font-mono text-white">{pct.toFixed(1)}%</span>}
        </div>
      )}
      <div className={`${height} bg-white/5 rounded-full overflow-hidden`}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
    </div>
  )
}

// ── Divider ────────────────────────────────────────────────────
export function Divider({ label }) {
  if (label) {
    return (
      <div className="flex items-center gap-3 my-4">
        <div className="flex-1 h-px bg-white/8" />
        <span className="text-xs text-slate-600 font-mono">{label}</span>
        <div className="flex-1 h-px bg-white/8" />
      </div>
    )
  }
  return <div className="h-px bg-white/8 my-4" />
}

// ── Tooltip wrapper ────────────────────────────────────────────
export function Tooltip({ children, text, position = 'top' }) {
  const pos = {
    top:    'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left:   'right-full top-1/2 -translate-y-1/2 mr-2',
    right:  'left-full top-1/2 -translate-y-1/2 ml-2',
  }[position]

  return (
    <div className="relative group inline-flex">
      {children}
      <div className={`absolute ${pos} z-50 px-2 py-1 bg-slate-700 border border-white/10
                       text-xs text-white rounded-lg whitespace-nowrap pointer-events-none
                       opacity-0 group-hover:opacity-100 transition-opacity duration-150`}>
        {text}
      </div>
    </div>
  )
}
