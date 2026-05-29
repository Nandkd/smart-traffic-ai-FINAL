// frontend/src/pages/SignalControl.jsx
// ML-based signal timing control for the Crossroad AI controller.
// Replaces the old 4-intersection demo panel.

import React, { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  SlidersHorizontal, Zap, RefreshCw, Clock, CheckCircle,
  ChevronUp, ChevronDown, ChevronLeft, ChevronRight,
  AlertTriangle, Brain, TrendingUp, Car, Bus, TruckIcon,
  Bike, Siren, BarChart2, Activity
} from 'lucide-react'
import api from '@/services/api'
import { predictionAPI } from '@/services/api'

const ROADS = ['north', 'south', 'east', 'west']
const ROAD_META = {
  north: { label: 'North Road', icon: ChevronUp,    color: '#3B82F6' },
  south: { label: 'South Road', icon: ChevronDown,  color: '#22C55E' },
  east:  { label: 'East Road',  icon: ChevronRight, color: '#F59E0B' },
  west:  { label: 'West Road',  icon: ChevronLeft,  color: '#8B5CF6' },
}
const SIGNAL_GLOW = {
  green:  { bg: '#22C55E', shadow: '0 0 14px #22C55E, 0 0 28px #22C55E55' },
  yellow: { bg: '#F59E0B', shadow: '0 0 14px #F59E0B' },
  red:    { bg: '#EF4444', shadow: '0 0 8px #EF444440' },
}

function TrafficBulb({ phase, size = 'sm' }) {
  const dim = size === 'lg' ? 'w-6 h-6' : 'w-3.5 h-3.5'
  return (
    <div className="flex flex-col gap-1 bg-slate-900 border border-white/15 rounded-lg p-1.5 items-center">
      {['red', 'yellow', 'green'].map(c => (
        <div key={c} className={`${dim} rounded-full transition-all duration-400`}
          style={{
            backgroundColor: phase === c ? SIGNAL_GLOW[c].bg : 'rgba(255,255,255,0.05)',
            boxShadow:        phase === c ? SIGNAL_GLOW[c].shadow : 'none',
          }} />
      ))}
    </div>
  )
}

// ── PCU bar for a road ─────────────────────────────────────────
function PCUBar({ road, pcu, maxPcu, isActive }) {
  const pct = maxPcu > 0 ? (pcu / maxPcu) * 100 : 0
  const meta = ROAD_META[road]
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs font-mono text-slate-400 w-10 capitalize">{road}</span>
      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
        <motion.div
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6 }}
          className="h-full rounded-full"
          style={{ backgroundColor: isActive ? '#22C55E' : meta.color }}
        />
      </div>
      <span className="text-xs font-mono text-white w-10 text-right">{pcu}</span>
    </div>
  )
}

// ── Road timing card ───────────────────────────────────────────
function RoadTimingCard({ road, state, timing, isActive, onChange }) {
  const meta = ROAD_META[road]
  const Icon = meta.icon
  const phase = state?.signal || 'red'
  return (
    <div className={`p-3 rounded-xl border transition-all ${
      isActive ? 'border-green-500/35 bg-green-500/6' : 'border-white/8 bg-white/2'
    }`}>
      <div className="flex items-center justify-between mb-2.5">
        <div className="flex items-center gap-2">
          <Icon className="w-3.5 h-3.5 flex-shrink-0" style={{ color: meta.color }} />
          <span className={`text-xs font-semibold capitalize ${isActive ? 'text-green-400' : 'text-slate-200'}`}>
            {road}
            {isActive && <span className="ml-1.5 text-green-500 font-mono text-xs">● green</span>}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <TrafficBulb phase={phase} />
          <div className="flex items-center gap-1">
            <button
              onClick={() => onChange(Math.max(0, timing - 5))}
              className="w-5 h-5 rounded border border-white/15 text-slate-400
                         hover:text-white hover:border-white/30 transition-colors
                         flex items-center justify-center text-xs font-bold">−</button>
            <span className={`text-sm font-bold font-mono w-10 text-center ${isActive ? 'text-green-400' : 'text-white'}`}>
              {timing}s
            </span>
            <button
              onClick={() => onChange(Math.min(120, timing + 5))}
              className="w-5 h-5 rounded border border-white/15 text-slate-400
                         hover:text-white hover:border-white/30 transition-colors
                         flex items-center justify-center text-xs font-bold">+</button>
          </div>
        </div>
      </div>
      <input
        type="range" min={0} max={120} step={5} value={timing}
        onChange={e => onChange(+e.target.value)}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
        style={{ accentColor: isActive ? '#22C55E' : meta.color }}
      />
      <div className="flex justify-between text-xs text-slate-700 mt-1 font-mono">
        <span>0s</span><span>60s</span><span>120s</span>
      </div>
      {state?.total_vehicles > 0 && (
        <p className="text-xs text-slate-500 mt-1.5 font-mono">
          {state.total_vehicles} veh · PCU {state.pcu_count}
          {state.density_class && state.density_class !== 'unknown' && (
            <span className={`ml-2 font-bold ${
              state.density_class === 'high' ? 'text-red-400'
              : state.density_class === 'medium' ? 'text-amber-400'
              : 'text-green-400'
            }`}>{state.density_class.toUpperCase()}</span>
          )}
        </p>
      )}
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────
export default function SignalControl() {
  const [crossroad, setCrossroad]   = useState(null)
  const [timings, setTimings]       = useState({ north: 30, south: 30, east: 30, west: 30 })
  const [activeRoad, setActiveRoad] = useState('north')
  const [applying, setApplying]     = useState(false)
  const [saved, setSaved]           = useState(false)
  const [optimizing, setOptimizing] = useState(false)
  const [optResult, setOptResult]   = useState(null)
  const pollRef = React.useRef(null)

  const fetchCrossroad = useCallback(async () => {
    try {
      const { data } = await api.get('/crossroad/state')
      setCrossroad(data)
      const roads = data.roads || {}
      // Sync timings from live state
      const updated = {}
      for (const r of ROADS) {
        const dur = roads[r]?.green_duration
        updated[r] = dur > 0 ? dur : timings[r]
      }
      setTimings(prev => ({ ...prev, ...updated }))
      const green = ROADS.find(r => roads[r]?.signal === 'green')
      if (green) setActiveRoad(green)
    } catch (err) { console.error(err) }
  }, [])  // eslint-disable-line

  useEffect(() => {
    fetchCrossroad()
    pollRef.current = setInterval(fetchCrossroad, 3000)
    return () => clearInterval(pollRef.current)
  }, [fetchCrossroad])

  const roads = crossroad?.roads || {}
  const signalMode = crossroad?.signal_mode || 'idle'
  const activeRoadLive = crossroad?.active_road

  const maxPcu = Math.max(...ROADS.map(r => roads[r]?.pcu_count || 0), 1)

  const handleApply = async () => {
    setApplying(true)
    try {
      await api.post('/crossroad/signal/timings', { ...timings, active_road: activeRoad })
      setSaved(true)
      setTimeout(() => setSaved(false), 2500)
      fetchCrossroad()
    } catch (err) { console.error(err) }
    finally { setApplying(false) }
  }

  const handleResumeAI = async () => {
    try {
      await api.post('/crossroad/signal/auto')
      fetchCrossroad()
    } catch (err) { console.error(err) }
  }

  const handleMLOptimize = async () => {
    setOptimizing(true)
    try {
      const lane_counts = {}
      for (const r of ROADS) {
        lane_counts[r] = roads[r]?.total_vehicles || 0
      }
      const emergencyLane = ROADS.find(r => roads[r]?.ambulance_detected)
      const { data } = await predictionAPI.optimizeSignal({
        lane_counts,
        ambulance_lane: emergencyLane || undefined,
      })
      const rec = data.recommended_timings
      setTimings(rec)
      // Set active road to highest timing
      const best = Object.entries(rec).reduce((a, b) => b[1] > a[1] ? b : a)[0]
      setActiveRoad(best)
      setOptResult(data)
    } catch (err) { console.error(err) }
    finally { setOptimizing(false) }
  }

  return (
    <div className="p-5 space-y-5 max-w-[1300px] mx-auto">

      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="font-display font-bold text-2xl text-white">Signal Control Center</h1>
          <p className="text-xs text-slate-500 mt-0.5 font-mono">
            Crossroad AI — ML-optimized adaptive timing · YOLOv11
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`text-xs font-mono px-3 py-1.5 rounded-xl border ${
            signalMode === 'emergency' ? 'text-red-400 border-red-500/30 bg-red-500/10 animate-pulse'
            : signalMode === 'manual'  ? 'text-amber-400 border-amber-500/30 bg-amber-500/10'
            : signalMode === 'auto'    ? 'text-green-400 border-green-500/30 bg-green-500/10'
            : 'text-slate-500 border-white/10 bg-white/3'
          }`}>
            {signalMode === 'emergency' ? '🚨 EMERGENCY'
              : signalMode === 'manual' ? '🖐 MANUAL'
              : signalMode === 'auto'   ? '⚡ AI AUTO'
              : '○ IDLE'}
          </span>
          <button onClick={handleResumeAI}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs
                       border border-green-500/25 text-green-400 bg-green-500/8
                       hover:bg-green-500/15 transition-colors">
            <Zap className="w-3 h-3" /> Return to AI Auto
          </button>
          <button onClick={fetchCrossroad}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs
                       border border-white/10 text-slate-400 hover:text-white
                       hover:border-white/25 transition-colors">
            <RefreshCw className="w-3 h-3" /> Refresh
          </button>
        </div>
      </div>

      {/* Emergency banner */}
      <AnimatePresence>
        {ROADS.some(r => roads[r]?.ambulance_detected) && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="flex items-center gap-3 p-3 rounded-xl bg-red-600/15 border border-red-500/30"
          >
            <Siren className="w-5 h-5 text-red-400 animate-pulse flex-shrink-0" />
            <div>
              <p className="text-sm font-bold text-red-300">
                🚨 AMBULANCE PRIORITY — {crossroad?.emergency_road?.toUpperCase()} ROAD CLEARED
              </p>
              <p className="text-xs text-red-400/70 mt-0.5">
                90s green on priority road · All other roads 0s · System in emergency mode
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main 2-col layout */}
      <div className="grid lg:grid-cols-5 gap-5">

        {/* Left: ML optimizer + active road selector */}
        <div className="lg:col-span-2 space-y-4">

          {/* ML optimize panel */}
          <div className="glass-card p-4 space-y-4">
            <div className="flex items-center gap-2">
              <Brain className="w-4 h-4 text-violet-400" />
              <h3 className="font-display font-semibold text-sm text-white">ML Signal Optimizer</h3>
            </div>

            <div className="space-y-1.5">
              <p className="text-xs font-mono text-slate-500 uppercase">
                Current lane vehicle counts (from Crossroad AI)
              </p>
              {ROADS.map(r => (
                <div key={r} className="flex items-center justify-between px-2 py-1.5
                                        bg-white/3 rounded-lg border border-white/5">
                  <div className="flex items-center gap-1.5">
                    {React.createElement(ROAD_META[r].icon, {
                      className: 'w-3 h-3',
                      style: { color: ROAD_META[r].color },
                    })}
                    <span className="text-xs text-slate-300 capitalize">{r}</span>
                  </div>
                  <div className="flex items-center gap-3 text-xs font-mono">
                    <span className="text-slate-400">{roads[r]?.total_vehicles || 0} veh</span>
                    <span className="text-amber-400">PCU {roads[r]?.pcu_count || 0}</span>
                  </div>
                </div>
              ))}
            </div>

            <button
              onClick={handleMLOptimize}
              disabled={optimizing}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl
                         bg-violet-600 hover:bg-violet-700 disabled:opacity-60
                         text-white text-xs font-semibold transition-colors">
              {optimizing
                ? <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                : <Zap className="w-3.5 h-3.5" />}
              Run ML Optimization
            </button>

            <AnimatePresence>
              {optResult && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="overflow-hidden"
                >
                  <div className="pt-3 border-t border-white/8 space-y-1">
                    <p className="text-xs font-mono text-violet-400 mb-2">
                      ✓ Optimized · Est. wait reduction: {optResult.estimated_wait_reduction_pct}%
                    </p>
                    {ROADS.map(r => (
                      <div key={r} className="flex justify-between text-xs">
                        <span className="text-slate-400 capitalize">{r}</span>
                        <span className="font-mono text-white">{optResult.recommended_timings?.[r]}s</span>
                      </div>
                    ))}
                    <p className="text-xs text-slate-600 mt-1 font-mono">
                      Cycle: {optResult.total_cycle_seconds}s total · {optResult.mode}
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Active road selector */}
          <div className="glass-card p-4 space-y-3">
            <p className="text-xs font-mono text-slate-500 uppercase">Active Green Road</p>
            <div className="grid grid-cols-2 gap-2">
              {ROADS.map(road => {
                const meta = ROAD_META[road]
                const Icon = meta.icon
                const isLive = road === activeRoadLive
                const isSelected = road === activeRoad
                return (
                  <button key={road} onClick={() => setActiveRoad(road)}
                    className={`flex items-center gap-2 p-2.5 rounded-xl border text-left
                                transition-all duration-200 ${
                      isSelected
                        ? 'border-green-500/50 bg-green-500/10'
                        : 'border-white/10 hover:border-white/20 hover:bg-white/3'
                    }`}
                  >
                    <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
                      style={{ backgroundColor: meta.color + '20', border: `1px solid ${meta.color}40` }}>
                      <Icon className="w-3.5 h-3.5" style={{ color: meta.color }} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className={`text-xs font-semibold capitalize truncate ${
                        isSelected ? 'text-green-400' : 'text-white'
                      }`}>{road}</p>
                      <p className="text-xs text-slate-500 font-mono">{timings[road]}s</p>
                    </div>
                    <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                      isSelected ? 'bg-green-400 shadow-[0_0_5px_#22C55E]'
                      : isLive ? 'bg-green-500/40'
                      : 'bg-slate-700'
                    }`} />
                  </button>
                )
              })}
            </div>
          </div>

          {/* PCU comparison */}
          {Object.values(roads).some(r => r?.pcu_count > 0) && (
            <div className="glass-card p-4 space-y-3">
              <p className="text-xs font-mono text-slate-500 uppercase">Road PCU Comparison</p>
              {ROADS.map(r => (
                <PCUBar key={r} road={r}
                  pcu={roads[r]?.pcu_count || 0}
                  maxPcu={maxPcu}
                  isActive={r === activeRoadLive}
                />
              ))}
              <p className="text-xs text-slate-600 font-mono">
                PCU = Passenger Car Unit (IRC:106-1990 standard)
              </p>
            </div>
          )}
        </div>

        {/* Right: Timing sliders + apply */}
        <div className="lg:col-span-3 space-y-4">
          <div className="glass-card p-5 space-y-4">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-amber-500/15 border border-amber-500/30
                              flex items-center justify-center">
                <SlidersHorizontal className="w-4 h-4 text-amber-400" />
              </div>
              <div>
                <h2 className="font-display font-bold text-sm text-white">Timing Sliders</h2>
                <p className="text-xs text-slate-500 font-mono">Green duration per road (0–120s)</p>
              </div>
            </div>

            <div className="space-y-3">
              {ROADS.map(road => (
                <RoadTimingCard
                  key={road}
                  road={road}
                  state={roads[road]}
                  timing={timings[road]}
                  isActive={activeRoad === road}
                  onChange={val => setTimings(p => ({ ...p, [road]: val }))}
                />
              ))}
            </div>

            <button
              onClick={handleApply}
              disabled={applying}
              className={`w-full py-3 rounded-xl text-sm font-semibold transition-all
                          flex items-center justify-center gap-2 ${
                saved    ? 'bg-green-600 text-white'
                : 'bg-amber-600 hover:bg-amber-700 disabled:opacity-60 text-white'
              }`}
            >
              {applying ? (
                <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Applying…</>
              ) : saved ? (
                <><CheckCircle className="w-4 h-4" /> Timings Applied!</>
              ) : (
                <><Clock className="w-4 h-4" /> Apply — {ROAD_META[activeRoad]?.label} Active</>
              )}
            </button>

            <p className="text-xs text-slate-600 text-center">
              Applies manual mode. "Return to AI Auto" restores PCU-based automatic control.
            </p>
          </div>

          {/* Live crossroad summary stats */}
          {crossroad && (
            <div className="glass-card p-4 space-y-3">
              <p className="text-xs font-mono text-slate-500 uppercase">Live Crossroad Summary</p>
              <div className="grid grid-cols-3 gap-3 text-center">
                {[
                  { label: 'Active Road', value: activeRoadLive ? activeRoadLive.toUpperCase() : '—', color: 'text-green-400' },
                  { label: 'Cycle Count', value: crossroad.cycle_count || 0, color: 'text-blue-400' },
                  { label: 'Signal Mode', value: signalMode.toUpperCase(), color: signalMode === 'emergency' ? 'text-red-400' : signalMode === 'manual' ? 'text-amber-400' : 'text-violet-400' },
                ].map(item => (
                  <div key={item.label} className="p-2.5 bg-white/3 rounded-xl border border-white/5">
                    <p className={`font-display font-bold text-lg ${item.color}`}>{item.value}</p>
                    <p className="text-xs text-slate-500 mt-0.5">{item.label}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

    </div>
  )
}
