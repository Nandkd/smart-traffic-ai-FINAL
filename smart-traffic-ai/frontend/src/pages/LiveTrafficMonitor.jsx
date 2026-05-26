// frontend/src/pages/LiveTrafficMonitor.jsx
// 4-intersection live monitor with auto signals, YOLOv8/v11, ambulance priority

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Siren, Zap, RefreshCw, Radio, Car, Bus,
  TruckIcon, CircleDot, AlertTriangle, ChevronUp,
  ChevronDown, ChevronLeft, ChevronRight, Activity
} from 'lucide-react'
import api from '@/services/api'

// ── Constants ────────────────────────────────────────────────
const INTERSECTIONS = [1, 2, 3, 4]
const SIGNAL_COLORS = {
  green:  { bg: '#22C55E', glow: 'shadow-green-500/60' },
  yellow: { bg: '#F59E0B', glow: 'shadow-amber-500/60' },
  red:    { bg: '#EF4444', glow: 'shadow-red-500/60' },
}
const DENSITY_COLOR = {
  low:    '#22C55E',
  medium: '#F59E0B',
  high:   '#EF4444',
}
const LANE_ICONS = {
  north: ChevronUp,
  south: ChevronDown,
  east:  ChevronRight,
  west:  ChevronLeft,
}

// ── Traffic Light Component ──────────────────────────────────
function TrafficLight({ phase, size = 'md' }) {
  const s = size === 'sm' ? 'w-3 h-3' : 'w-5 h-5'
  const lights = ['red', 'yellow', 'green']
  return (
    <div className="flex flex-col items-center gap-1 bg-slate-900 border border-white/10
                    rounded-lg p-1.5">
      {lights.map(color => (
        <div key={color}
          className={`${s} rounded-full transition-all duration-300`}
          style={{
            backgroundColor: phase === color
              ? SIGNAL_COLORS[color].bg
              : 'rgba(255,255,255,0.08)',
            boxShadow: phase === color
              ? `0 0 10px ${SIGNAL_COLORS[color].bg}`
              : 'none',
          }}
        />
      ))}
    </div>
  )
}

// ── Lane Signal Display ───────────────────────────────────────
function LaneSignals({ signals, laneCounts, activeLane, remaining }) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {Object.entries(signals).map(([lane, phase]) => {
        const Icon = LANE_ICONS[lane]
        const count = laneCounts?.[lane] || 0
        const isActive = lane === activeLane
        return (
          <div key={lane}
            className={`flex items-center gap-2 p-2 rounded-xl border transition-all duration-300 ${
              isActive
                ? 'border-green-500/40 bg-green-500/10'
                : phase === 'red'
                ? 'border-red-500/20 bg-red-500/5'
                : 'border-amber-500/30 bg-amber-500/8'
            }`}>
            <TrafficLight phase={phase} size="sm" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1">
                <Icon className="w-3 h-3 text-slate-400" />
                <span className="text-xs font-mono text-slate-300 capitalize">{lane}</span>
              </div>
              <div className="flex items-center gap-1 mt-0.5">
                <span className="text-xs font-bold text-white">{count}</span>
                <span className="text-xs text-slate-500">vehicles</span>
              </div>
            </div>
            {isActive && (
              <div className="text-right">
                <span className="text-xs font-mono font-bold text-green-400">
                  {Math.round(remaining)}s
                </span>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── Intersection Card ─────────────────────────────────────────
function IntersectionCard({ state, model, onAmbulance, onClearAmb, onOverride }) {
  const [showOverride, setShowOverride] = useState(false)
  const [overrideLane, setOverrideLane] = useState('north')
  const [overrideDur, setOverrideDur]   = useState(30)

  if (!state) return (
    <div className="glass-card p-4 flex items-center justify-center h-64">
      <div className="w-6 h-6 border-2 border-crimson-500/30 border-t-crimson-500
                      rounded-full animate-spin" />
    </div>
  )

  const isEmergency = state.ambulance_detected
  const density     = state.density_class || 'low'

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className={`glass-card p-4 flex flex-col gap-3 border transition-all duration-500 ${
        isEmergency
          ? 'border-red-500/50 bg-red-500/5 shadow-lg shadow-red-500/10'
          : 'border-white/8'
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-display font-semibold text-sm text-white truncate">
            {state.name}
          </h3>
          <div className="flex items-center gap-2 mt-1">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            <span className="text-xs text-slate-500 font-mono">
              {model?.toUpperCase()} · {state.fps || 30} FPS · {state.inference_ms}ms
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="text-xs font-mono font-bold px-2 py-0.5 rounded-full border"
            style={{
              color: DENSITY_COLOR[density],
              borderColor: DENSITY_COLOR[density] + '40',
              backgroundColor: DENSITY_COLOR[density] + '15',
            }}
          >
            {density.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Ambulance alert */}
      <AnimatePresence>
        {isEmergency && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="flex items-center justify-between p-2 rounded-xl
                       bg-red-600/15 border border-red-500/30"
          >
            <div className="flex items-center gap-2">
              <Siren className="w-4 h-4 text-red-400 animate-pulse" />
              <div>
                <p className="text-xs font-bold text-red-300">AMBULANCE PRIORITY</p>
                <p className="text-xs text-red-400/70 font-mono">
                  {state.ambulance_lane?.toUpperCase()} lane · 90s green
                </p>
              </div>
            </div>
            <button
              onClick={() => onClearAmb(state.intersection_id)}
              className="text-xs text-slate-400 hover:text-white border border-white/10
                         hover:border-white/30 px-2 py-1 rounded-lg transition-colors"
            >
              Clear
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Simulated camera feed */}
      <div className="relative bg-slate-900 rounded-xl overflow-hidden aspect-video
                      border border-white/5">
        {/* Grid overlay */}
        <div className="absolute inset-0 bg-grid-pattern opacity-10" />

        {/* Scanner line */}
        {!isEmergency && (
          <motion.div
            animate={{ y: ['0%', '100%', '0%'] }}
            transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
            className="absolute inset-x-0 h-px bg-gradient-to-r
                       from-transparent via-green-400 to-transparent opacity-40 z-10"
          />
        )}
        {isEmergency && (
          <motion.div
            animate={{ opacity: [0.3, 0.8, 0.3] }}
            transition={{ duration: 0.8, repeat: Infinity }}
            className="absolute inset-0 bg-red-600/10 z-10"
          />
        )}

        {/* Vehicle dots */}
        <div className="absolute inset-0 p-2">
          {(state.detections || []).slice(0, 12).map((det, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.05 }}
              className="absolute w-2 h-1.5 rounded-sm"
              style={{
                left:  `${10 + (i % 6) * 15}%`,
                top:   `${15 + Math.floor(i / 6) * 45}%`,
                backgroundColor: det.class === 'ambulance'
                  ? '#EF4444'
                  : det.class === 'bus' ? '#8B5CF6'
                  : det.class === 'truck' ? '#06B6D4'
                  : '#3B82F6',
              }}
            />
          ))}
        </div>

        {/* HUD overlay */}
        <div className="absolute bottom-0 inset-x-0 p-2 bg-gradient-to-t
                        from-black/70 to-transparent">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-white/80">
              {state.total_vehicles} vehicles
            </span>
            <span className="text-xs font-mono"
              style={{ color: DENSITY_COLOR[density] }}>
              ● {density}
            </span>
          </div>
        </div>

        {/* Emergency overlay text */}
        {isEmergency && (
          <div className="absolute inset-0 flex items-center justify-center z-20">
            <motion.p
              animate={{ opacity: [1, 0.4, 1] }}
              transition={{ duration: 0.6, repeat: Infinity }}
              className="text-red-400 font-mono font-bold text-xs bg-black/60
                         px-3 py-1 rounded-full border border-red-500/40"
            >
              🚨 EMERGENCY CLEARANCE
            </motion.p>
          </div>
        )}
      </div>

      {/* Lane signals */}
      <LaneSignals
        signals={state.signals || {}}
        laneCounts={state.lane_counts || {}}
        activeLane={state.active_lane}
        remaining={state.remaining_seconds || 0}
      />

      {/* Vehicle type breakdown */}
      <div className="flex gap-2 flex-wrap">
        {Object.entries(state.vehicle_counts || {}).map(([type, count]) => (
          <span key={type}
            className="text-xs font-mono px-2 py-0.5 rounded-full
                       bg-white/5 border border-white/10 text-slate-300">
            {type}: {count}
          </span>
        ))}
      </div>

      {/* Action buttons */}
      <div className="flex gap-2 flex-wrap">
        {/* Ambulance button */}
        {!isEmergency ? (
          <button
            onClick={() => onAmbulance(state.intersection_id)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs
                       font-medium bg-red-600/10 border border-red-500/25
                       text-red-400 hover:bg-red-600/20 transition-colors"
          >
            <Siren className="w-3 h-3" /> Ambulance
          </button>
        ) : (
          <button
            onClick={() => onClearAmb(state.intersection_id)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs
                       font-medium bg-green-500/10 border border-green-500/25
                       text-green-400 hover:bg-green-500/20 transition-colors"
          >
            <RefreshCw className="w-3 h-3" /> Clear Emergency
          </button>
        )}

        {/* Override button */}
        <button
          onClick={() => setShowOverride(!showOverride)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs
                     font-medium border border-white/10 text-slate-400
                     hover:text-white hover:border-white/25 transition-colors"
        >
          <Zap className="w-3 h-3" /> Override
        </button>
      </div>

      {/* Manual override panel */}
      <AnimatePresence>
        {showOverride && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden border-t border-white/8 pt-3 space-y-2"
          >
            <p className="text-xs text-slate-500 font-mono">MANUAL OVERRIDE</p>
            <div className="flex gap-2">
              <select
                value={overrideLane}
                onChange={e => setOverrideLane(e.target.value)}
                className="flex-1 bg-slate-800 border border-white/10 rounded-lg
                           px-2 py-1.5 text-xs text-white focus:outline-none"
              >
                {['north','south','east','west'].map(l => (
                  <option key={l} value={l}>{l.toUpperCase()}</option>
                ))}
              </select>
              <input
                type="number" min={10} max={90} value={overrideDur}
                onChange={e => setOverrideDur(parseInt(e.target.value))}
                className="w-16 bg-slate-800 border border-white/10 rounded-lg
                           px-2 py-1.5 text-xs text-white font-mono
                           focus:outline-none text-center"
              />
              <span className="text-xs text-slate-500 self-center">s</span>
            </div>
            <button
              onClick={() => {
                onOverride(state.intersection_id, overrideLane, overrideDur)
                setShowOverride(false)
              }}
              className="w-full py-1.5 rounded-lg bg-crimson-600 hover:bg-crimson-700
                         text-white text-xs font-medium transition-colors"
            >
              Apply Override
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Metrics row */}
      <div className="flex items-center justify-between text-xs text-slate-600
                      font-mono border-t border-white/5 pt-2">
        <span>Cycles: {state.cycle_count}</span>
        <span>Cleared: {state.vehicles_cleared}</span>
        <span>Score: {(state.congestion_score * 100).toFixed(0)}%</span>
      </div>
    </motion.div>
  )
}

// ── Main Page ─────────────────────────────────────────────────
export default function LiveTrafficMonitor() {
  const [states, setStates]       = useState({})
  const [summary, setSummary]     = useState(null)
  const [model, setModel]         = useState('yolov8')
  const [switching, setSwitching] = useState(false)
  const [loading, setLoading]     = useState(true)
  const [autoMode, setAutoMode]   = useState(true)
  const intervalRef = useRef(null)

  // Fetch all states
  const fetchStates = useCallback(async () => {
    try {
      const [stateRes, summaryRes] = await Promise.all([
        api.get('/live/states'),
        api.get('/live/summary'),
      ])
      const stateMap = {}
      for (const s of stateRes.data.intersections || []) {
        stateMap[s.intersection_id] = s
      }
      setStates(stateMap)
      setSummary(summaryRes.data)
    } catch (err) {
      console.error('Fetch error:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStates()
    intervalRef.current = setInterval(fetchStates, 1500)
    return () => clearInterval(intervalRef.current)
  }, [fetchStates])

  // Switch model
  const switchModel = async (m) => {
    setSwitching(true)
    try {
      await api.post('/live/model/switch', { model: m })
      setModel(m)
    } catch (err) { console.error(err) }
    finally { setSwitching(false) }
  }

  // Ambulance trigger
  const triggerAmbulance = async (iid) => {
    const lanes = ['north','south','east','west']
    const lane  = lanes[Math.floor(Math.random() * lanes.length)]
    try {
      await api.post(`/live/ambulance/${iid}`, { lane })
    } catch (err) { console.error(err) }
  }

  // Clear ambulance
  const clearAmbulance = async (iid) => {
    try {
      await api.post(`/live/ambulance/${iid}/clear`)
    } catch (err) { console.error(err) }
  }

  // Signal override
  const overrideSignal = async (iid, lane, duration) => {
    try {
      await api.post(`/live/signal/${iid}/override`, { lane, duration })
    } catch (err) { console.error(err) }
  }

  const anyEmergency = Object.values(states).some(s => s?.ambulance_detected)

  return (
    <div className="p-4 space-y-4 max-w-[1600px] mx-auto">

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="font-display font-bold text-2xl text-white">
            Live Traffic Monitor
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            4 intersections · Auto signal control · Real-time detection
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Model switcher */}
          <div className="flex items-center gap-1 bg-slate-800/60 border border-white/10
                          rounded-xl p-1">
            {['yolov8','yolov11'].map(m => (
              <button
                key={m}
                onClick={() => switchModel(m)}
                disabled={switching}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold
                            transition-all duration-200 ${
                  model === m
                    ? 'bg-crimson-600 text-white shadow-lg'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {m.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Auto mode toggle */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Auto AI</span>
            <button
              onClick={() => setAutoMode(!autoMode)}
              className={`w-10 h-5 rounded-full transition-colors relative ${
                autoMode ? 'bg-crimson-600' : 'bg-white/10'
              }`}
            >
              <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full
                               transition-transform ${
                autoMode ? 'translate-x-5' : 'translate-x-0.5'
              }`} />
            </button>
          </div>

          {/* Refresh */}
          <button onClick={fetchStates}
            className="btn-ghost py-1.5 px-3 text-xs flex items-center gap-1.5">
            <RefreshCw className="w-3.5 h-3.5" /> Refresh
          </button>
        </div>
      </div>

      {/* Emergency banner */}
      <AnimatePresence>
        {anyEmergency && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex items-center gap-3 p-3 rounded-xl bg-red-600/15
                       border border-red-500/30"
          >
            <Siren className="w-5 h-5 text-red-400 animate-pulse flex-shrink-0" />
            <div>
              <p className="text-sm font-bold text-red-300">
                AMBULANCE PRIORITY ACTIVE
              </p>
              <p className="text-xs text-red-400/70">
                Emergency vehicles detected — signals automatically cleared
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Summary stats */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            {
              label: 'Total Vehicles',
              value: summary.total_vehicles_live,
              color: 'text-blue-400',
              icon: Car,
            },
            {
              label: 'Avg FPS',
              value: `${summary.avg_fps} FPS`,
              color: 'text-green-400',
              icon: Activity,
            },
            {
              label: 'Avg Congestion',
              value: `${(summary.avg_congestion_score * 100).toFixed(0)}%`,
              color: 'text-amber-400',
              icon: Radio,
            },
            {
              label: 'Emergencies',
              value: summary.ambulance_active_count,
              color: 'text-red-400',
              icon: Siren,
            },
          ].map((item, i) => {
            const Icon = item.icon
            return (
              <motion.div
                key={item.label}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.06 }}
                className="glass-card px-4 py-3 flex items-center gap-3"
              >
                <Icon className={`w-4 h-4 ${item.color} flex-shrink-0`} />
                <div>
                  <p className={`font-display font-bold text-lg ${item.color}`}>
                    {item.value}
                  </p>
                  <p className="text-xs text-slate-500">{item.label}</p>
                </div>
              </motion.div>
            )
          })}
        </div>
      )}

      {/* 4-intersection grid */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="w-10 h-10 border-2 border-crimson-500/30
                          border-t-crimson-500 rounded-full animate-spin" />
        </div>
      ) : (
        <div className="grid md:grid-cols-2 xl:grid-cols-2 gap-4">
          {INTERSECTIONS.map(iid => (
            <IntersectionCard
              key={iid}
              state={states[iid]}
              model={model}
              onAmbulance={triggerAmbulance}
              onClearAmb={clearAmbulance}
              onOverride={overrideSignal}
            />
          ))}
        </div>
      )}

      {/* How auto signals work */}
      <div className="glass-card p-5">
        <h3 className="font-display font-semibold text-sm text-white mb-3 flex items-center gap-2">
          <Zap className="w-4 h-4 text-amber-400" />
          How Auto AI Signal Control Works
        </h3>
        <div className="grid md:grid-cols-4 gap-4 text-xs text-slate-400">
          {[
            {
              step: '01',
              title: 'Detect',
              desc: `${model.toUpperCase()} scans each intersection every 2 seconds counting vehicles per lane`,
            },
            {
              step: '02',
              title: 'Predict',
              desc: 'ML model computes optimal green duration based on vehicle count, hour, and day',
            },
            {
              step: '03',
              title: 'Decide',
              desc: 'Highest-density lane gets priority. Starvation prevention ensures no lane waits > 120s',
            },
            {
              step: '04',
              title: 'Override',
              desc: 'Ambulance CNN triggers immediate 90s green clearance on the emergency lane',
            },
          ].map(item => (
            <div key={item.step} className="flex gap-3">
              <span className="font-mono text-crimson-500 font-bold text-base
                               flex-shrink-0">{item.step}</span>
              <div>
                <p className="font-semibold text-white mb-1">{item.title}</p>
                <p className="leading-relaxed">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Timing formula */}
        <div className="mt-4 p-3 bg-white/3 rounded-xl border border-white/8
                        font-mono text-xs text-slate-400">
          <p className="text-slate-300 mb-1">Green Duration Formula:</p>
          <p>vehicles ≤ 5  → 10s &nbsp;|&nbsp;
             ≤ 20 → 20s &nbsp;|&nbsp;
             ≤ 40 → 30s &nbsp;|&nbsp;
             ≤ 70 → 50s &nbsp;|&nbsp;
             &gt; 70 → 75s
          </p>
          <p className="mt-1 text-amber-400">
            Ambulance detected → immediate 90s green · all others 5s red
          </p>
        </div>
      </div>
    </div>
  )
}
