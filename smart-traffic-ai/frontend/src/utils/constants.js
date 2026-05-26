// frontend/src/utils/constants.js
/**
 * Application-wide constants.
 */

// ── API ────────────────────────────────────────────────────────
export const API_BASE_URL  = '/api'
export const API_TIMEOUT   = 30_000   // 30 s

// ── Polling intervals (ms) ─────────────────────────────────────
export const DASHBOARD_POLL_MS  = 15_000
export const MONITOR_POLL_MS    =  2_000
export const SIGNAL_POLL_MS     = 10_000

// ── Traffic classes ────────────────────────────────────────────
export const DENSITY_CLASSES = ['low', 'medium', 'high']

export const DENSITY_COLORS = {
  low:    '#22C55E',
  medium: '#F59E0B',
  high:   '#EF4444',
}

export const DENSITY_BG = {
  low:    'rgba(34,197,94,0.08)',
  medium: 'rgba(245,158,11,0.08)',
  high:   'rgba(239,68,68,0.08)',
}

// ── Vehicle classes ────────────────────────────────────────────
export const VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'ambulance']

export const VEHICLE_COLORS = {
  car:        '#3B82F6',
  motorcycle: '#F59E0B',
  bus:        '#8B5CF6',
  truck:      '#06B6D4',
  ambulance:  '#EF4444',
}

// ── Chart colours (ordered palette) ───────────────────────────
export const CHART_PALETTE = [
  '#EF4444', '#3B82F6', '#22C55E',
  '#F59E0B', '#8B5CF6', '#06B6D4',
  '#EC4899', '#84CC16',
]

// ── Lane directions ────────────────────────────────────────────
export const LANES = ['north', 'south', 'east', 'west']

export const LANE_COLORS = {
  north: '#3B82F6',
  south: '#8B5CF6',
  east:  '#22C55E',
  west:  '#F59E0B',
}

// ── Signal states ──────────────────────────────────────────────
export const SIGNAL_STATUS = {
  active:    { label: 'Active',    color: '#22C55E' },
  emergency: { label: 'Emergency', color: '#EF4444' },
  manual:    { label: 'Manual',    color: '#F59E0B' },
  offline:   { label: 'Offline',   color: '#64748b' },
}

// ── Signal timing (seconds) ────────────────────────────────────
export const MIN_GREEN_SEC = 5
export const MAX_GREEN_SEC = 120
export const DEFAULT_GREEN_SEC = 30
export const EMERGENCY_GREEN_SEC = 90

// ── User roles ─────────────────────────────────────────────────
export const USER_ROLES = ['admin', 'operator', 'viewer']

// ── Days of week ───────────────────────────────────────────────
export const DAY_NAMES  = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
export const DAY_SHORT  = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

// ── Navigation routes ──────────────────────────────────────────
export const ROUTES = {
  HOME:       '/',
  LOGIN:      '/login',
  DASHBOARD:  '/dashboard',
  MONITOR:    '/monitor',
  ANALYTICS:  '/analytics',
  PREDICT:    '/predict',
  AMBULANCE:  '/ambulance',
  SIGNALS:    '/signals',
  ADMIN:      '/admin',
}

// ── Model metadata ─────────────────────────────────────────────
export const MODELS = {
  yolo: {
    name: 'YOLOv8s Traffic Detector',
    accuracy: '89.1% mAP@0.5',
    latency: '12ms',
    classes: VEHICLE_CLASSES,
  },
  cnn: {
    name: 'CNN Ambulance Classifier',
    accuracy: '96.2%',
    latency: '8ms',
    classes: ['non_ambulance', 'ambulance'],
  },
  ensemble: {
    name: 'Voting Ensemble (RF + XGB)',
    accuracy: '96.4%',
    latency: '<1ms',
    classes: DENSITY_CLASSES,
  },
}
