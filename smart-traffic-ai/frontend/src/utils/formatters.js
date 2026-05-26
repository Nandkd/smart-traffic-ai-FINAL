// frontend/src/utils/formatters.js
/**
 * Shared formatting and helper utilities for the UI.
 */

// ── Numbers ────────────────────────────────────────────────────

/** Format a number with thousands separator. */
export const formatNumber = (n) =>
  typeof n === 'number' ? n.toLocaleString() : '—'

/** Format a float as a percentage string. */
export const formatPct = (n, decimals = 1) =>
  typeof n === 'number' ? `${(n * 100).toFixed(decimals)}%` : '—'

/** Format bytes to KB/MB. */
export const formatBytes = (bytes) => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(2)} MB`
}

// ── Time ───────────────────────────────────────────────────────

/** Format an ISO timestamp to a readable local time string. */
export const formatTime = (iso) => {
  if (!iso) return '—'
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/** Format an ISO timestamp to a date string. */
export const formatDate = (iso) => {
  if (!iso) return '—'
  return new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric' })
}

/** Relative time from now (e.g. "3 minutes ago"). */
export const timeAgo = (iso) => {
  if (!iso) return '—'
  const diff = (Date.now() - new Date(iso)) / 1000
  if (diff < 60)   return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

/** Format seconds duration to mm:ss or hh:mm:ss. */
export const formatDuration = (seconds) => {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${m}:${String(s).padStart(2, '0')}`
}

// ── Traffic domain helpers ─────────────────────────────────────

/** Map a density class string to a Tailwind color class. */
export const densityColor = (cls) => ({
  low:    'text-green-400',
  medium: 'text-amber-400',
  high:   'text-crimson-400',
}[cls] || 'text-slate-400')

/** Map a density class string to a hex color. */
export const densityHex = (cls) => ({
  low:    '#22C55E',
  medium: '#F59E0B',
  high:   '#EF4444',
}[cls] || '#94A3B8')

/** Map a density class string to a badge class. */
export const densityBadge = (cls) => ({
  low:    'badge-low',
  medium: 'badge-medium',
  high:   'badge-high',
}[cls] || 'badge-low')

/** Given a congestion score (0–1), return the density class. */
export const scoreToDensity = (score) => {
  if (score >= 0.65) return 'high'
  if (score >= 0.35) return 'medium'
  return 'low'
}

/** Map signal status to display label + color. */
export const signalStatusMeta = (status) => ({
  active:    { label: 'Active',    color: 'text-green-400',  bg: 'bg-green-500/10' },
  emergency: { label: 'Emergency', color: 'text-crimson-400', bg: 'bg-crimson-500/10' },
  manual:    { label: 'Manual',    color: 'text-amber-400',  bg: 'bg-amber-500/10' },
  offline:   { label: 'Offline',   color: 'text-slate-400',  bg: 'bg-slate-700/30' },
}[status] || { label: status, color: 'text-slate-400', bg: 'bg-slate-700/30' })

// ── Arrays / objects ───────────────────────────────────────────

/** Clamp a number between min and max. */
export const clamp = (n, min, max) => Math.min(Math.max(n, min), max)

/** Interpolate between two values. */
export const lerp = (a, b, t) => a + (b - a) * t

/** Group array of objects by a key. */
export const groupBy = (arr, key) =>
  arr.reduce((acc, item) => {
    const k = item[key]
    ;(acc[k] = acc[k] || []).push(item)
    return acc
  }, {})

/** Pick specific keys from an object. */
export const pick = (obj, keys) =>
  keys.reduce((acc, k) => (obj[k] !== undefined ? { ...acc, [k]: obj[k] } : acc), {})

/** Generate a random hex color. */
export const randomColor = () =>
  `#${Math.floor(Math.random() * 0xFFFFFF).toString(16).padStart(6, '0')}`
