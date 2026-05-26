// frontend/src/pages/SignalControl.jsx
import React, { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { TrafficCone, Zap, AlertTriangle, RefreshCw, Timer, ArrowUp, ArrowDown, ArrowLeft, ArrowRight } from 'lucide-react'
import { signalsAPI, predictionAPI } from '@/services/api'

const LANE_ICONS = { north: ArrowUp, south: ArrowDown, east: ArrowRight, west: ArrowLeft }
const LANE_COLORS = { north: '#3B82F6', south: '#8B5CF6', east: '#22C55E', west: '#F59E0B' }

function TimingRing({ value, max = 90, color, label }) {
  const r = 36
  const circumference = 2 * Math.PI * r
  const pct = Math.min(value / max, 1)
  const dashOffset = circumference * (1 - pct)

  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="relative w-24 h-24">
        <svg className="w-full h-full -rotate-90">
          <circle cx="48" cy="48" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="6" />
          <motion.circle
            cx="48" cy="48" r={r} fill="none"
            stroke={color} strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={circumference}
            animate={{ strokeDashoffset: dashOffset }}
            transition={{ duration: 0.5 }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-display font-bold text-xl text-white">{value}s</span>
        </div>
      </div>
      <span className="text-xs text-slate-400 capitalize">{label}</span>
    </div>
  )
}

function SignalCard({ signal, onUpdate, onEmergency, onReset }) {
  const [editing, setEditing] = useState(false)
  const [tempTimings, setTempTimings] = useState(signal.timings)
  const [laneCounts, setLaneCounts] = useState({ north: 20, south: 15, east: 25, west: 10 })
  const [optimizing, setOptimizing] = useState(false)

  const isEmergency = signal.status === 'emergency'

  const handleOptimize = async () => {
    setOptimizing(true)
    try {
      const { data } = await predictionAPI.optimizeSignal({
        lane_counts: laneCounts,
        ambulance_lane: isEmergency ? signal.emergency_lane : undefined,
      })
      const t = data.recommended_timings
      await onUpdate(signal.id, { timings: t })
      setTempTimings(t)
    } catch (err) { console.error(err) }
    finally { setOptimizing(false) }
  }

  const handleSave = async () => {
    await onUpdate(signal.id, { timings: tempTimings })
    setEditing(false)
  }

  return (
    <div className={`glass-card p-5 ${isEmergency ? 'border-crimson-600/40 bg-crimson-600/5' : ''}`}>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <TrafficCone className={`w-4 h-4 ${isEmergency ? 'text-crimson-400' : 'text-green-400'}`} />
            <h3 className="font-display font-semibold text-sm text-white">{signal.location_name}</h3>
          </div>
          <p className="text-xs text-slate-500 font-mono">
            {signal.latitude?.toFixed(4)}, {signal.longitude?.toFixed(4)}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {isEmergency
            ? <span className="badge-emergency">EMERGENCY</span>
            : <span className="badge-low">ACTIVE</span>
          }
        </div>
      </div>

      {/* Timing rings */}
      <div className="grid grid-cols-4 gap-2 mb-5">
        {['north','south','east','west'].map(lane => {
          const Icon = LANE_ICONS[lane]
          return (
            <div key={lane}>
              <TimingRing
                value={signal.timings[lane]}
                color={isEmergency && signal.emergency_lane === lane ? '#EF4444' : LANE_COLORS[lane]}
                label={lane}
              />
            </div>
          )
        })}
      </div>

      {/* ML lane count inputs for optimizer */}
      <div className="mb-4 p-3 bg-white/3 rounded-xl border border-white/6">
        <p className="text-xs text-slate-500 mb-2 font-mono">VEHICLE COUNT / LANE (for ML optimizer)</p>
        <div className="grid grid-cols-4 gap-2">
          {['north','south','east','west'].map(lane => (
            <div key={lane}>
              <label className="text-xs text-slate-500 capitalize block mb-1">{lane}</label>
              <input
                type="number" min={0} max={200} value={laneCounts[lane]}
                onChange={e => setLaneCounts(prev => ({ ...prev, [lane]: parseInt(e.target.value) || 0 }))}
                className="w-full bg-slate-800 border border-white/10 rounded-lg px-2 py-1.5 text-xs text-white font-mono
                           focus:outline-none focus:border-crimson-500/50"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={handleOptimize}
          disabled={optimizing}
          className="btn-primary py-2 px-3 text-xs flex items-center gap-1.5 disabled:opacity-60"
        >
          {optimizing
            ? <span className="w-3 h-3 border border-white/30 border-t-white rounded-full animate-spin" />
            : <Zap className="w-3 h-3" />
          }
          ML Optimize
        </button>

        <button
          onClick={() => onEmergency(signal.id, 'north')}
          className="py-2 px-3 text-xs flex items-center gap-1.5 bg-crimson-600/15 border border-crimson-600/30
                     text-crimson-400 rounded-xl hover:bg-crimson-600/25 transition-colors"
        >
          <AlertTriangle className="w-3 h-3" /> Emergency
        </button>

        {isEmergency && (
          <button
            onClick={() => onReset(signal.id)}
            className="py-2 px-3 text-xs flex items-center gap-1.5 bg-green-500/10 border border-green-500/20
                       text-green-400 rounded-xl hover:bg-green-500/20 transition-colors"
          >
            <RefreshCw className="w-3 h-3" /> Reset
          </button>
        )}

        <button
          onClick={() => setEditing(!editing)}
          className="btn-ghost py-2 px-3 text-xs flex items-center gap-1.5 ml-auto"
        >
          <Timer className="w-3 h-3" /> Manual
        </button>
      </div>

      {/* Manual edit panel */}
      <AnimatePresence>
        {editing && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-4 pt-4 border-t border-white/8">
              <p className="text-xs text-slate-500 mb-3 font-mono">MANUAL TIMING (seconds)</p>
              <div className="grid grid-cols-2 gap-3">
                {['north','south','east','west'].map(lane => (
                  <div key={lane}>
                    <label className="text-xs text-slate-400 capitalize block mb-1">{lane}</label>
                    <input
                      type="number" min={5} max={120} value={tempTimings[lane]}
                      onChange={e => setTempTimings(prev => ({ ...prev, [lane]: parseInt(e.target.value) || 5 }))}
                      className="w-full bg-slate-800 border border-white/10 rounded-xl px-3 py-2 text-sm text-white font-mono
                                 focus:outline-none focus:border-crimson-500/50"
                    />
                  </div>
                ))}
              </div>
              <div className="flex gap-2 mt-3">
                <button onClick={handleSave} className="btn-primary py-2 px-4 text-xs flex-1">Save</button>
                <button onClick={() => setEditing(false)} className="btn-ghost py-2 px-4 text-xs flex-1">Cancel</button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function SignalControl() {
  const [signals, setSignals] = useState([])
  const [loading, setLoading] = useState(true)

  const fetchSignals = useCallback(async () => {
    try {
      const { data } = await signalsAPI.getAll()
      setSignals(data.signals || [])
    } catch (err) { console.error(err) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => {
    fetchSignals()
    const interval = setInterval(fetchSignals, 10000)
    return () => clearInterval(interval)
  }, [fetchSignals])

  const handleUpdate = async (id, payload) => {
    try {
      const { data } = await signalsAPI.update(id, payload)
      setSignals(prev => prev.map(s => s.id === id ? data.signal : s))
    } catch (err) { console.error(err) }
  }

  const handleEmergency = async (id, lane) => {
    try {
      const { data } = await signalsAPI.emergency(id, lane)
      setSignals(prev => prev.map(s => s.id === id ? data.signal : s))
    } catch (err) { console.error(err) }
  }

  const handleReset = async (id) => {
    try {
      const { data } = await signalsAPI.reset(id)
      setSignals(prev => prev.map(s => s.id === id ? data.signal : s))
    } catch (err) { console.error(err) }
  }

  if (loading) return (
    <div className="flex items-center justify-center h-full">
      <div className="w-8 h-8 border-2 border-crimson-500/30 border-t-crimson-500 rounded-full animate-spin" />
    </div>
  )

  const emergencyCount = signals.filter(s => s.status === 'emergency').length

  return (
    <div className="p-6 space-y-6 max-w-[1300px] mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="font-display font-bold text-2xl text-white">Signal Control Center</h1>
          <p className="text-sm text-slate-500 mt-0.5">ML-optimized adaptive signal timing · 4 intersections</p>
        </div>
        <div className="flex items-center gap-3">
          {emergencyCount > 0 && (
            <div className="flex items-center gap-2 px-3 py-2 bg-crimson-600/15 border border-crimson-600/30 rounded-xl text-crimson-400 text-xs font-mono animate-pulse">
              <AlertTriangle className="w-3.5 h-3.5" />
              {emergencyCount} EMERGENCY ACTIVE
            </div>
          )}
          <button onClick={fetchSignals}
            className="btn-ghost py-2 px-4 text-sm flex items-center gap-2">
            <RefreshCw className="w-3.5 h-3.5" /> Refresh
          </button>
        </div>
      </div>

      {/* Legend */}
      <div className="glass-card p-4 flex items-center gap-6 flex-wrap text-xs">
        <p className="text-slate-500 font-mono">LANE COLORS:</p>
        {Object.entries(LANE_COLORS).map(([lane, color]) => (
          <div key={lane} className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-slate-400 capitalize">{lane}</span>
          </div>
        ))}
        <div className="ml-auto text-slate-500">
          Cycle: auto-refreshes every 10s
        </div>
      </div>

      {/* Signal cards grid */}
      <div className="grid md:grid-cols-2 gap-5">
        {signals.map((sig, i) => (
          <motion.div
            key={sig.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08 }}
          >
            <SignalCard
              signal={sig}
              onUpdate={handleUpdate}
              onEmergency={handleEmergency}
              onReset={handleReset}
            />
          </motion.div>
        ))}
      </div>

      {/* Global actions */}
      <div className="glass-card p-5">
        <h3 className="font-display font-semibold text-sm text-white mb-4">Global Controls</h3>
        <div className="flex gap-3 flex-wrap">
          <button
            onClick={() => signals.forEach(s => handleReset(s.id))}
            className="btn-ghost py-2 px-5 text-sm flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" /> Reset All Signals
          </button>
          <button
            onClick={() => signals.forEach(s => handleEmergency(s.id, 'north'))}
            className="py-2 px-5 text-sm flex items-center gap-2 bg-crimson-600/10 border border-crimson-600/25
                       text-crimson-400 rounded-xl hover:bg-crimson-600/20 transition-colors"
          >
            <AlertTriangle className="w-4 h-4" /> Emergency All (North Priority)
          </button>
        </div>
        <p className="text-xs text-slate-600 mt-3">
          ML Optimize uses lane vehicle counts + time context to compute optimal green durations per the trained prediction model.
        </p>
      </div>
    </div>
  )
}
