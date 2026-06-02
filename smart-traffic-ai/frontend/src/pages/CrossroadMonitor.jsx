// frontend/src/pages/CrossroadMonitor.jsx
// To swap uploaded videos for live RTSP: replace videoUrl objectURL with stream src string.
// All other code stays identical — CCTVFeed only depends on the src prop.

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  RotateCcw, Siren, Car, Bus,
  TruckIcon, Bike, AlertTriangle, CheckCircle,
  ChevronUp, ChevronDown, ChevronLeft, ChevronRight,
  Film, MapPin, Radio, Activity,
} from 'lucide-react'
import api from '@/services/api'

// ── Constants ────────────────────────────────────────────────
const ROADS = ['north', 'south', 'east', 'west']

const ROAD_META = {
  north: { label: 'North Road',  icon: ChevronUp,    color: '#3B82F6', position: 'top'    },
  south: { label: 'South Road',  icon: ChevronDown,  color: '#22C55E', position: 'bottom' },
  east:  { label: 'East Road',   icon: ChevronRight, color: '#F59E0B', position: 'right'  },
  west:  { label: 'West Road',   icon: ChevronLeft,  color: '#8B5CF6', position: 'left'   },
}

const VEHICLE_META = {
  car:           { icon: Car,       color: '#3B82F6', label: 'Cars'          },
  motorcycle:    { icon: Bike,      color: '#F59E0B', label: 'Motorcycles'   },
  auto_rickshaw: { icon: Car,       color: '#F97316', label: 'Auto Rickshaw' },
  bus:           { icon: Bus,       color: '#8B5CF6', label: 'Buses'         },
  truck:         { icon: TruckIcon, color: '#06B6D4', label: 'Trucks'        },
  bicycle:       { icon: Bike,      color: '#6EE7B7', label: 'Bicycles'      },
  pedestrian:    { icon: Car,       color: '#94A3B8', label: 'Pedestrians'   },
  ambulance:     { icon: Siren,     color: '#EF4444', label: 'Ambulance'     },
}

const SIGNAL_GLOW = {
  green:  { bg: '#22C55E', glow: '0 0 20px #22C55E, 0 0 40px #22C55E55' },
  yellow: { bg: '#F59E0B', glow: '0 0 20px #F59E0B' },
  red:    { bg: '#EF4444', glow: '0 0 12px #EF444455' },
}

const DENSITY_STYLE = {
  low:     { text: 'text-green-400', bg: 'bg-green-500/10', border: 'border-green-500/30' },
  medium:  { text: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30' },
  high:    { text: 'text-red-400',   bg: 'bg-red-500/10',   border: 'border-red-500/30'  },
  unknown: { text: 'text-slate-400', bg: 'bg-white/3',      border: 'border-white/8'     },
}

const PHASES = {
  A: { label: 'Phase A', movement: 'N+S Straight',    roads: ['north','south'], arrows: '↑↓', color: '#3B82F6',
       desc: 'North & South straight lanes move simultaneously. East & West stopped.' },
  B: { label: 'Phase B', movement: 'N+S Right Turns', roads: ['north','south'], arrows: '↗↙', color: '#8B5CF6',
       desc: 'North & South dedicated right-turn arrow. Prevents head-on collisions.' },
  C: { label: 'Phase C', movement: 'E+W Straight',    roads: ['east','west'],   arrows: '→←', color: '#F59E0B',
       desc: 'East & West straight lanes move simultaneously. North & South stopped.' },
  D: { label: 'Phase D', movement: 'E+W Right Turns', roads: ['east','west'],   arrows: '↘↖', color: '#22C55E',
       desc: 'East & West dedicated right-turn arrow. Prevents head-on collisions.'  },
}

const ALLOWED = ['.mp4','.avi','.mov','.mkv','.webm','.wmv']

// ── Live clock hook ──────────────────────────────────────────
function useLiveClock() {
  const [time, setTime] = useState(() => new Date().toLocaleTimeString('en-GB'))
  useEffect(() => {
    const t = setInterval(() => setTime(new Date().toLocaleTimeString('en-GB')), 1000)
    return () => clearInterval(t)
  }, [])
  return time
}

// ── Simulated AI bounding boxes ──────────────────────────────
// Positions are seeded from vehicle counts so they stay stable between renders.
// Replace with real YOLO bbox coords when RTSP is connected.
function DetectionBoxes({ vehicleCounts }) {
  const boxes = useMemo(() => {
    const result = []
    const total = Object.values(vehicleCounts || {}).reduce((a, b) => a + b, 0)
    if (!total) return result

    let seed = total * 31337
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff
      return (seed >>> 0) / 0xffffffff
    }

    Object.entries(vehicleCounts || {})
      .filter(([, v]) => v > 0)
      .forEach(([cls, count]) => {
        const color = VEHICLE_META[cls]?.color || '#3B82F6'
        const n = Math.min(count, 3)
        for (let i = 0; i < n; i++) {
          const w = 13 + rand() * 18
          const h = 9  + rand() * 13
          const x = 3  + rand() * (82 - w)
          const y = 5  + rand() * (78 - h)
          result.push({ x, y, w, h, color, label: cls.replace('_', ' ') })
        }
      })
    return result.slice(0, 10)
  }, [vehicleCounts])

  if (!boxes.length) return null

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {boxes.map((box, i) => (
        <motion.div
          key={i}
          className="absolute"
          initial={{ opacity: 0 }}
          animate={{ opacity: [0.55, 0.95, 0.55] }}
          transition={{ duration: 2.2 + i * 0.25, repeat: Infinity, delay: i * 0.12 }}
          style={{
            left: `${box.x}%`, top: `${box.y}%`,
            width: `${box.w}%`, height: `${box.h}%`,
            border: `1.5px solid ${box.color}`,
            boxShadow: `0 0 6px ${box.color}55`,
          }}
        >
          {/* label tag */}
          <div
            className="absolute -top-[14px] left-0 text-[7px] font-mono font-bold
                       px-1 py-px leading-none whitespace-nowrap"
            style={{ background: box.color + 'cc', color: '#000' }}
          >
            {box.label}
          </div>
          {/* corner marks */}
          {[
            'absolute top-0 left-0 w-2 h-2 border-t-2 border-l-2',
            'absolute top-0 right-0 w-2 h-2 border-t-2 border-r-2',
            'absolute bottom-0 left-0 w-2 h-2 border-b-2 border-l-2',
            'absolute bottom-0 right-0 w-2 h-2 border-b-2 border-r-2',
          ].map((cls, j) => (
            <div key={j} className={cls} style={{ borderColor: box.color }} />
          ))}
        </motion.div>
      ))}

      {/* Horizontal scanner sweep */}
      <motion.div
        className="absolute left-0 right-0 h-px pointer-events-none"
        style={{ background: 'linear-gradient(90deg,transparent,rgba(34,197,94,0.5),transparent)' }}
        animate={{ top: ['8%', '92%', '8%'] }}
        transition={{ duration: 4.5, repeat: Infinity, ease: 'linear' }}
      />
    </div>
  )
}

// ── CCTVFeed ─────────────────────────────────────────────────
// videoUrl: objectURL from File | RTSP/HLS stream URL (future)
function CCTVFeed({ road, videoUrl, state }) {
  const meta     = ROAD_META[road]
  const signal   = state?.signal || 'red'
  const density  = state?.density_class || 'unknown'
  const vehicles = state?.total_vehicles || 0
  const counts   = state?.vehicle_counts || {}
  const isEmerg  = state?.ambulance_detected
  const ds       = DENSITY_STYLE[density] || DENSITY_STYLE.unknown
  const time     = useLiveClock()
  const date     = new Date().toLocaleDateString('en-GB').replace(/\//g, '-')
  const camId    = `CAM-${road[0].toUpperCase()}${(road.charCodeAt(0) * 17 % 89) + 10}`
  const videoRef = useRef(null)

  useEffect(() => {
    if (videoRef.current && videoUrl) {
      videoRef.current.load()
      videoRef.current.play().catch(() => {})
    }
  }, [videoUrl])

  const sc = {
    green:  { dot: 'bg-green-400',  text: 'text-green-400',  border: 'border-green-500/40',  glow: '0 0 10px #22C55E80' },
    yellow: { dot: 'bg-amber-400',  text: 'text-amber-400',  border: 'border-amber-500/40',  glow: '0 0 10px #F59E0B80' },
    red:    { dot: 'bg-red-400',    text: 'text-red-400',    border: 'border-red-500/40',    glow: '0 0 10px #EF444480' },
  }[signal] || { dot: 'bg-red-400', text: 'text-red-400', border: 'border-red-500/40', glow: '' }

  return (
    <div
      className={`relative overflow-hidden rounded-xl bg-black aspect-video
                  border transition-all duration-500
                  ${isEmerg ? 'border-red-500/70' : signal === 'green' ? 'border-green-500/30' : 'border-white/10'}`}
      style={isEmerg ? { boxShadow: '0 0 24px rgba(239,68,68,0.35)' }
           : signal === 'green' ? { boxShadow: '0 0 16px rgba(34,197,94,0.2)' } : {}}
    >
      {/* Video */}
      {videoUrl ? (
        <video ref={videoRef} src={videoUrl} autoPlay muted loop playsInline
          className="w-full h-full object-cover" />
      ) : (
        <div className="w-full h-full flex flex-col items-center justify-center bg-slate-900/80 gap-2">
          <Film className="w-6 h-6 text-slate-700" />
          <p className="text-[10px] text-slate-600 font-mono">NO SIGNAL · {camId}</p>
        </div>
      )}

      {/* Overlays: dark vignette + gradient bars */}
      <div className="absolute inset-0 pointer-events-none"
        style={{ background: 'radial-gradient(ellipse at center, transparent 45%, rgba(0,0,0,0.65) 100%)' }} />
      <div className="absolute inset-x-0 top-0 h-10 pointer-events-none"
        style={{ background: 'linear-gradient(to bottom, rgba(0,0,0,0.75), transparent)' }} />
      <div className="absolute inset-x-0 bottom-0 h-10 pointer-events-none"
        style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.75), transparent)' }} />
      {/* Scanlines texture */}
      <div className="absolute inset-0 pointer-events-none opacity-[0.025]"
        style={{ backgroundImage: 'repeating-linear-gradient(0deg,#fff 0px,#fff 1px,transparent 1px,transparent 3px)' }} />

      {/* ── Top-left: camera ID + road name ── */}
      <div className="absolute top-1.5 left-2 flex flex-col gap-0.5 z-10">
        <div className="flex items-center gap-1">
          <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: meta.color }} />
          <span className="text-[10px] font-mono font-bold text-white drop-shadow">{camId}</span>
        </div>
        <span className="text-[9px] font-mono text-white/60">{meta.label.toUpperCase()}</span>
      </div>

      {/* ── Top-right: REC + date/time ── */}
      <div className="absolute top-1.5 right-2 flex flex-col items-end gap-0.5 z-10">
        {videoUrl && (
          <div className="flex items-center gap-1">
            <motion.div animate={{ opacity: [1, 0.15, 1] }} transition={{ duration: 1.1, repeat: Infinity }}
              className="w-1.5 h-1.5 rounded-full bg-red-500" />
            <span className="text-[10px] font-mono font-bold text-red-400">REC</span>
          </div>
        )}
        <span className="text-[9px] font-mono text-white/60">{date}</span>
        <span className="text-[10px] font-mono font-bold text-white">{time}</span>
      </div>

      {/* ── AI bounding boxes ── */}
      {videoUrl && vehicles > 0 && <DetectionBoxes vehicleCounts={counts} />}

      {/* ── Bottom-left: vehicle count + density ── */}
      <div className="absolute bottom-1.5 left-2 flex flex-col gap-0.5 z-10">
        {vehicles > 0 && (
          <div className="flex items-center gap-1">
            <Car className="w-2.5 h-2.5 text-white/70" />
            <span className="text-[10px] font-mono font-bold text-white">{vehicles} vehicles</span>
          </div>
        )}
        {density !== 'unknown' && (
          <span className={`text-[8px] font-mono font-bold px-1 py-px rounded
                            bg-black/60 border ${ds.border} ${ds.text}`}>
            {density.toUpperCase()}
          </span>
        )}
      </div>

      {/* ── Bottom-right: signal status ── */}
      <div className="absolute bottom-1.5 right-2 flex items-center gap-1.5 z-10">
        <span className={`text-[8px] font-mono font-bold px-1.5 py-px rounded
                          bg-black/60 border ${sc.border} ${sc.text}`}
          style={{ boxShadow: sc.glow }}>
          {signal.toUpperCase()}
        </span>
        <div className={`w-2 h-2 rounded-full ${sc.dot}`} style={{ boxShadow: sc.glow }} />
      </div>

      {/* ── Bottom-center: AI ACTIVE badge ── */}
      {videoUrl && (
        <div className="absolute bottom-1.5 left-1/2 -translate-x-1/2 z-10">
          <span className="text-[7px] font-mono text-green-400/70 bg-black/50 px-1.5 py-px rounded">
            AI ACTIVE
          </span>
        </div>
      )}

      {/* ── Emergency red flash ── */}
      {isEmerg && (
        <motion.div
          animate={{ opacity: [0, 0.28, 0] }}
          transition={{ duration: 0.75, repeat: Infinity }}
          className="absolute inset-0 bg-red-600/25 pointer-events-none z-20"
        />
      )}
    </div>
  )
}

// ── Traffic Light ────────────────────────────────────────────
function TrafficLight({ phase, size = 'md' }) {
  const dim = size === 'lg' ? 'w-8 h-8' : size === 'md' ? 'w-5 h-5' : 'w-3 h-3'
  return (
    <div className="flex flex-col gap-1.5 bg-slate-900 border border-white/15 rounded-xl p-2 items-center shadow-xl">
      {['red','yellow','green'].map(c => (
        <div key={c} className={`${dim} rounded-full transition-all duration-500`}
          style={{
            backgroundColor: phase === c ? SIGNAL_GLOW[c].bg : 'rgba(255,255,255,0.05)',
            boxShadow: phase === c ? SIGNAL_GLOW[c].glow : 'none',
          }} />
      ))}
    </div>
  )
}

// ── Phase Indicator ──────────────────────────────────────────
function PhaseIndicator({ currentPhase, phaseScores, signalMode }) {
  if (!currentPhase || signalMode === 'emergency' || signalMode === 'idle') return null
  const ph = PHASES[currentPhase]
  if (!ph) return null
  const maxScore = Math.max(...Object.values(phaseScores || {}), 1)

  return (
    <motion.div key={currentPhase} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
      className="rounded-xl border p-3 space-y-2"
      style={{ borderColor: ph.color + '40', backgroundColor: ph.color + '0d' }}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-2xl font-mono font-black" style={{ color: ph.color }}>{ph.arrows}</span>
          <div>
            <p className="text-sm font-bold text-white">{ph.label}</p>
            <p className="text-xs font-mono" style={{ color: ph.color }}>{ph.movement}</p>
          </div>
        </div>
        <div className="flex gap-1">
          {ph.roads.map(r => (
            <span key={r} className="text-xs font-mono px-1.5 py-0.5 rounded-md bg-white/8 text-white capitalize">{r}</span>
          ))}
        </div>
      </div>
      <p className="text-xs text-slate-400 leading-relaxed">{ph.desc}</p>
      {phaseScores && (
        <div className="space-y-1 pt-1 border-t border-white/8">
          <p className="text-xs text-slate-500 font-mono">PHASE SCORES (PCU)</p>
          {Object.entries(PHASES).map(([key, p]) => {
            const score    = phaseScores[key] || 0
            const pct      = maxScore > 0 ? (score / maxScore) * 100 : 0
            const isActive = key === currentPhase
            return (
              <div key={key} className="flex items-center gap-2">
                <span className={`text-xs font-mono w-6 font-bold ${isActive ? 'text-white' : 'text-slate-500'}`}>{key}</span>
                <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                  <motion.div animate={{ width: `${pct}%` }} transition={{ duration: 0.6 }}
                    className="h-full rounded-full"
                    style={{ backgroundColor: isActive ? p.color : p.color + '60' }} />
                </div>
                <span className={`text-xs font-mono w-8 text-right ${isActive ? 'text-white' : 'text-slate-500'}`}>{score}</span>
              </div>
            )
          })}
        </div>
      )}
    </motion.div>
  )
}

// ── Crossroad Visual ─────────────────────────────────────────
function CrossroadVisual({ roads, currentPhase, emergencyRoad, signalMode }) {
  const activeRoads = currentPhase ? (PHASES[currentPhase]?.roads || []) : []

  return (
    <div className="relative w-full aspect-square max-w-[280px] mx-auto select-none">
      <div className="absolute inset-y-[38%] inset-x-0 bg-slate-700/60 border-y border-white/10" />
      <div className="absolute inset-x-[38%] inset-y-0 bg-slate-700/60 border-x border-white/10" />
      <div className="absolute inset-[38%] bg-slate-600/80 border border-white/15 flex items-center justify-center z-10">
        {currentPhase && signalMode !== 'emergency'
          ? <span className="text-xs font-black text-white font-mono">{currentPhase}</span>
          : <div className="w-3 h-3 rounded-full bg-white/20" />}
      </div>
      {/* Dashed lane lines */}
      <div className="absolute left-1/2 top-0 bottom-[38%] w-px border-l-2 border-dashed border-white/20 -translate-x-1/2" />
      <div className="absolute left-1/2 top-[62%] bottom-0 w-px border-l-2 border-dashed border-white/20 -translate-x-1/2" />
      <div className="absolute top-1/2 left-0 right-[38%] h-px border-t-2 border-dashed border-white/20 -translate-y-1/2" />
      <div className="absolute top-1/2 left-[62%] right-0 h-px border-t-2 border-dashed border-white/20 -translate-y-1/2" />

      {ROADS.map(road => {
        const meta     = ROAD_META[road]
        const state    = roads[road]
        const phase    = state?.signal || 'red'
        const isActive = activeRoads.includes(road)
        const isEmerg  = road === emergencyRoad
        const vc       = state?.total_vehicles || 0
        const posClass = {
          top:    'top-1 left-1/2 -translate-x-1/2 flex-col items-center',
          bottom: 'bottom-1 left-1/2 -translate-x-1/2 flex-col-reverse items-center',
          left:   'left-1 top-1/2 -translate-y-1/2 flex-row items-center',
          right:  'right-1 top-1/2 -translate-y-1/2 flex-row-reverse items-center',
        }[meta.position]

        return (
          <div key={road} className={`absolute flex gap-2 z-20 ${posClass}`}>
            <TrafficLight phase={phase} size="sm" />
            <div className="text-center">
              <p className="text-xs font-mono font-bold"
                style={{ color: isEmerg ? '#EF4444' : isActive ? '#22C55E' : '#64748b' }}>
                {road.toUpperCase()}
              </p>
              {vc > 0 && <p className="text-xs font-bold text-white">{vc}</p>}
            </div>
          </div>
        )
      })}

      {emergencyRoad && (
        <motion.div animate={{ opacity: [0.2, 0.5, 0.2] }} transition={{ duration: 0.8, repeat: Infinity }}
          className="absolute inset-0 rounded-full border-4 border-red-500/40 z-30" />
      )}

      <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 whitespace-nowrap">
        <span className={`text-xs font-mono px-3 py-1 rounded-full border ${
          signalMode === 'emergency' ? 'text-red-400 border-red-500/30 bg-red-500/10' :
          signalMode === 'auto'      ? 'text-green-400 border-green-500/30 bg-green-500/10' :
          signalMode === 'manual'    ? 'text-amber-400 border-amber-500/30 bg-amber-500/10' :
          'text-slate-500 border-white/10 bg-white/3'
        }`}>
          {signalMode === 'emergency' ? '🚨 EMERGENCY' :
           signalMode === 'auto'      ? '⚡ AUTO AI' :
           signalMode === 'manual'    ? '🖐 MANUAL' : '○ IDLE'}
        </span>
      </div>
    </div>
  )
}

// ── Timeline ─────────────────────────────────────────────────
function Timeline({ frames }) {
  if (!frames?.length) return null
  const max = Math.max(...frames.map(f => f.total_vehicles), 1)
  return (
    <div className="flex items-end gap-px h-8 rounded-lg overflow-hidden bg-white/3 p-1">
      {frames.map((f, i) => {
        const h = Math.max(10, (f.total_vehicles / max) * 100)
        const c = f.density_class === 'high' ? '#EF4444' : f.density_class === 'medium' ? '#F59E0B' : '#22C55E'
        return (
          <div key={i} className="flex-1 rounded-sm min-w-0.5 transition-all"
            style={{ height: `${h}%`, backgroundColor: c }}
            title={`${f.timestamp_sec}s — ${f.total_vehicles} vehicles`} />
        )
      })}
    </div>
  )
}

// ── Road Panel ───────────────────────────────────────────────
function RoadPanel({ road, state, models, videoUrl, onUpload, onAmbulance, onClearAmb, onOverride, onReset }) {
  const [model, setModel]       = useState('yolov8s')
  const [drag, setDrag]         = useState(false)
  const [showOverride, setOver] = useState(false)
  const [overDur, setOverDur]   = useState(30)
  const [frames, setFrames]     = useState([])
  const fileRef = useRef(null)
  const meta    = ROAD_META[road]
  const Icon    = meta.icon
  const status  = state?.status || 'idle'
  const phase   = state?.signal || 'red'
  const density = state?.density_class || 'unknown'
  const ds      = DENSITY_STYLE[density] || DENSITY_STYLE.unknown
  const isEmerg = state?.ambulance_detected

  useEffect(() => {
    if (status === 'completed') {
      api.get(`/crossroad/frames/${road}`).then(r => setFrames(r.data.frames || [])).catch(() => {})
    }
  }, [status, road])

  const handleFile = (file) => {
    if (!file) return
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!ALLOWED.includes(ext)) { alert('Unsupported format. Use: ' + ALLOWED.join(', ')); return }
    onUpload(road, file, model)
  }

  return (
    <div className={`glass-card flex flex-col gap-3 p-4 border transition-all duration-500 ${
      isEmerg         ? 'border-red-500/50 bg-red-500/5'
      : phase === 'green' ? 'border-green-500/30 bg-green-500/3'
      : 'border-white/8'
    }`}>

      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl flex items-center justify-center border"
            style={{ borderColor: meta.color + '40', backgroundColor: meta.color + '15', color: meta.color }}>
            <Icon className="w-4 h-4" />
          </div>
          <div>
            <p className="font-display font-bold text-sm text-white">{meta.label}</p>
            <p className="text-xs text-slate-500 font-mono">
              {state?.model_used
                ? `${models[state.model_used]?.name || state.model_used} · ${state.real_inference ? 'Real YOLO' : 'OpenCV'}`
                : 'No video uploaded'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <TrafficLight phase={phase} size="sm" />
          {density !== 'unknown' && (
            <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded-full border ${ds.text} ${ds.bg} ${ds.border}`}>
              {density.toUpperCase()}
            </span>
          )}
        </div>
      </div>

      {/* CCTV Feed — always visible once video uploaded */}
      <CCTVFeed road={road} videoUrl={videoUrl} state={state} />

      {/* Ambulance alert */}
      <AnimatePresence>
        {isEmerg && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
            className="flex items-center justify-between p-2.5 rounded-xl bg-red-600/15 border border-red-500/30">
            <div className="flex items-center gap-2">
              <Siren className="w-4 h-4 text-red-400 animate-pulse flex-shrink-0" />
              <div>
                <p className="text-xs font-bold text-red-300">AMBULANCE — 90s PRIORITY</p>
                <p className="text-xs text-red-400/60 font-mono">
                  {state.ambulance_timestamps?.length > 0
                    ? `Detected at ${state.ambulance_timestamps.slice(0,3).map(t=>`${t}s`).join(', ')}`
                    : 'Manual trigger active'}
                </p>
              </div>
            </div>
            <button onClick={() => onClearAmb(road)}
              className="text-xs px-2 py-1 border border-white/15 rounded-lg text-slate-300 hover:text-white transition-colors flex-shrink-0">
              Clear
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* IDLE — model select + upload drop zone */}
      {status === 'idle' && (
        <div className="space-y-3">
          <div>
            <p className="text-xs font-mono text-slate-500 mb-1.5">YOLO MODEL</p>
            <select value={model} onChange={e => setModel(e.target.value)}
              className="w-full bg-slate-800 border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-crimson-500/40">
              <optgroup label="── YOLOv11 (Latest)">
                {['yolov11n','yolov11s','yolov11m'].map(k => (
                  <option key={k} value={k}>{models[k] ? `${models[k].name} — ${models[k].speed} · ${models[k].map}% mAP` : k}</option>
                ))}
              </optgroup>
              <optgroup label="── YOLOv8 ✅ Recommended">
                {['yolov8n','yolov8s','yolov8m','yolov8l'].map(k => (
                  <option key={k} value={k}>{models[k] ? `${models[k].name} — ${models[k].speed} · ${models[k].map}% mAP${models[k].rec ? ' ★' : ''}` : k}</option>
                ))}
              </optgroup>
              <optgroup label="── YOLOv9 / v10">
                {['yolov9c','yolov10s'].map(k => (
                  <option key={k} value={k}>{models[k] ? `${models[k].name} — ${models[k].speed}` : k}</option>
                ))}
              </optgroup>
            </select>
          </div>
          <div
            className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all duration-200 ${
              drag ? 'border-crimson-500/60 bg-crimson-500/8' : 'border-white/15 hover:border-white/30 hover:bg-white/3'
            }`}
            onDragOver={e => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={e => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]) }}
            onClick={() => fileRef.current?.click()}
          >
            <input ref={fileRef} type="file" accept="video/*" className="hidden"
              onChange={e => handleFile(e.target.files[0])} />
            <Film className={`w-8 h-8 mx-auto mb-2 transition-colors ${drag ? 'text-crimson-400' : 'text-slate-600'}`} />
            <p className="text-sm font-medium text-white mb-1">Drop {meta.label} video</p>
            <p className="text-xs text-slate-500">MP4 · AVI · MOV · MKV</p>
          </div>
        </div>
      )}

      {/* PROCESSING */}
      {(status === 'queued' || status === 'processing') && (
        <div className="space-y-3">
          <div>
            <div className="flex justify-between text-xs mb-1.5">
              <span className="text-slate-400 font-mono">
                {status === 'queued' ? 'Queued...' : `Processing frames (${state?.frames_done || 0} done)`}
              </span>
              <span className="text-white font-mono font-bold">{state?.progress || 0}%</span>
            </div>
            <div className="h-2 bg-white/5 rounded-full overflow-hidden">
              <motion.div animate={{ width: `${state?.progress || 0}%` }} transition={{ duration: 0.4 }}
                className="h-full rounded-full bg-gradient-to-r from-crimson-600 to-crimson-400" />
            </div>
          </div>
          {state?.current_frame && (
            <div className="p-2.5 bg-white/3 rounded-xl border border-white/5">
              <p className="text-xs text-slate-500 font-mono mb-1.5">LATEST FRAME</p>
              <div className="flex flex-wrap gap-2 mb-2">
                {Object.entries(state.current_frame.vehicle_counts || {}).map(([cls, cnt]) => (
                  <span key={cls} className="text-xs font-mono text-white">
                    <span className="text-slate-400">{cls}:</span> {cnt}
                  </span>
                ))}
              </div>
              {(state.current_frame.straight_count > 0 || state.current_frame.right_count > 0) && (
                <div className="flex gap-2 text-xs font-mono">
                  <span className="text-blue-400">↑ Straight: {state.current_frame.straight_count} ({state.current_frame.straight_pcu} PCU)</span>
                  <span className="text-amber-400">↗ Right: {state.current_frame.right_count} ({state.current_frame.right_pcu} PCU)</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* FAILED */}
      {status === 'failed' && (
        <div className="flex flex-col items-center gap-2 py-3">
          <AlertTriangle className="w-8 h-8 text-red-400" />
          <p className="text-sm text-red-300">Processing failed</p>
          <button onClick={() => onReset(road)} className="btn-ghost text-xs py-1.5 px-4">Reset &amp; Try Again</button>
        </div>
      )}

      {/* COMPLETED stats */}
      {(status === 'completed' || (status === 'processing' && state?.current_frame)) && (
        <div className="space-y-3">
          <div>
            <p className="text-xs text-slate-500 font-mono mb-2">DETECTED VEHICLES (avg/frame)</p>
            <div className="grid grid-cols-2 gap-1.5">
              {Object.entries(state?.vehicle_counts || {}).map(([cls, cnt]) => {
                const vm = VEHICLE_META[cls] || VEHICLE_META.car
                return (
                  <div key={cls} className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg bg-white/3 border border-white/5">
                    <vm.icon className="w-3.5 h-3.5 flex-shrink-0" style={{ color: vm.color }} />
                    <span className="text-xs text-slate-300 capitalize truncate flex-1">{cls.replace('_',' ')}</span>
                    <span className="text-xs font-bold text-white">{cnt}</span>
                  </div>
                )
              })}
            </div>
          </div>

          {(state?.straight_count > 0 || state?.right_count > 0) && (
            <div className="p-2.5 bg-white/3 rounded-xl border border-white/5">
              <p className="text-xs text-slate-500 font-mono mb-2">LANE SPLIT (auto 65/35 ROI)</p>
              <div className="grid grid-cols-2 gap-2 mb-2">
                <div className="text-center">
                  <p className="text-sm font-bold text-blue-400">{state.straight_count}</p>
                  <p className="text-xs text-slate-500">↑ Straight</p>
                  <p className="text-xs font-mono text-blue-300">{state.straight_pcu} PCU</p>
                </div>
                <div className="text-center">
                  <p className="text-sm font-bold text-amber-400">{state.right_count}</p>
                  <p className="text-xs text-slate-500">↗ Right Turn</p>
                  <p className="text-xs font-mono text-amber-300">{state.right_pcu} PCU</p>
                </div>
              </div>
              <div className="h-1.5 bg-white/5 rounded-full overflow-hidden flex">
                {(() => {
                  const total = (state.straight_count || 0) + (state.right_count || 0)
                  const pct   = total > 0 ? ((state.straight_count || 0) / total) * 100 : 65
                  return (<><div className="h-full bg-blue-500/70 transition-all" style={{ width: `${pct}%` }} /><div className="h-full bg-amber-500/70 flex-1" /></>)
                })()}
              </div>
            </div>
          )}

          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="p-2 bg-white/3 rounded-xl border border-white/5">
              <p className="text-base font-bold text-white">{state?.total_vehicles || 0}</p>
              <p className="text-xs text-slate-500">Avg vehicles</p>
            </div>
            <div className="p-2 bg-white/3 rounded-xl border border-white/5">
              <p className="text-base font-bold text-amber-400">{state?.pcu_count || 0}</p>
              <p className="text-xs text-slate-500">PCU score</p>
            </div>
            <div className={`p-2 rounded-xl border ${phase === 'green' ? 'bg-green-500/10 border-green-500/25' : 'bg-white/3 border-white/5'}`}>
              <p className={`text-base font-bold ${phase === 'green' ? 'text-green-400' : 'text-slate-400'}`}>
                {state?.green_duration || 0}s
              </p>
              <p className="text-xs text-slate-500">Green time</p>
            </div>
          </div>

          {frames.length > 0 && (
            <div>
              <p className="text-xs text-slate-500 font-mono mb-1">TRAFFIC TIMELINE — {frames.length} frames</p>
              <Timeline frames={frames} />
            </div>
          )}

          {state?.ambulance_timestamps?.length > 0 && (
            <div className="p-2 bg-red-500/8 rounded-xl border border-red-500/20">
              <p className="text-xs font-mono text-red-400 mb-1">🚨 AMBULANCE DETECTED</p>
              <p className="text-xs text-slate-400">At: {state.ambulance_timestamps.slice(0,5).map(t=>`${t}s`).join(' · ')}</p>
            </div>
          )}
        </div>
      )}

      {/* Green signal info */}
      {phase === 'green' && state?.green_duration > 0 && (
        <div className="flex items-center gap-2 p-2 bg-green-500/8 rounded-xl border border-green-500/20">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <p className="text-xs text-green-300 font-mono">GREEN — {state.green_duration}s · {state.total_vehicles} vehicles waiting</p>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2 flex-wrap border-t border-white/5 pt-3">
        {status !== 'idle' && status !== 'processing' && (
          <button onClick={() => onReset(road)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs border border-white/10 text-slate-400 hover:text-white hover:border-white/25 transition-colors">
            <RotateCcw className="w-3 h-3" /> New Video
          </button>
        )}
        {!isEmerg ? (
          <button onClick={() => onAmbulance(road)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs bg-red-600/10 border border-red-500/25 text-red-400 hover:bg-red-600/20 transition-colors">
            <Siren className="w-3 h-3" /> Ambulance
          </button>
        ) : (
          <button onClick={() => onClearAmb(road)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs bg-green-500/10 border border-green-500/25 text-green-400 hover:bg-green-500/20 transition-colors">
            <CheckCircle className="w-3 h-3" /> Clear
          </button>
        )}
        <button onClick={() => setOver(!showOverride)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs border border-white/10 text-slate-400 hover:text-white hover:border-white/25 transition-colors ml-auto">
          Override
        </button>
      </div>

      {/* Manual override panel */}
      <AnimatePresence>
        {showOverride && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden border-t border-white/8 pt-3 space-y-2">
            <p className="text-xs text-slate-500 font-mono">MANUAL OVERRIDE</p>
            <div className="flex items-center gap-2">
              <input type="number" min={5} max={120} value={overDur} onChange={e => setOverDur(+e.target.value)}
                className="w-20 bg-slate-800 border border-white/10 rounded-lg px-2 py-1.5 text-xs text-white font-mono focus:outline-none text-center" />
              <span className="text-xs text-slate-500">seconds green</span>
              <button onClick={() => { onOverride(road, overDur); setOver(false) }}
                className="flex-1 py-1.5 bg-crimson-600 hover:bg-crimson-700 text-white text-xs rounded-lg font-medium transition-colors">
                Apply
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── Active Feed Center Panel ─────────────────────────────────
// Shows the currently green/emergency road's video in the center column.
function ActiveFeedPanel({ roads, videoUrls, currentPhase, emergencyRoad }) {
  const activeRoad = useMemo(() => {
    if (emergencyRoad && videoUrls[emergencyRoad]) return emergencyRoad
    const green = ROADS.find(r => roads[r]?.signal === 'green' && videoUrls[r])
    if (green) return green
    return ROADS.find(r => videoUrls[r]) || null
  }, [roads, videoUrls, emergencyRoad])

  if (!activeRoad) return null

  const meta    = ROAD_META[activeRoad]
  const isEmerg = activeRoad === emergencyRoad

  return (
    <div className="glass-card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Radio className="w-3.5 h-3.5 text-green-400 animate-pulse" />
          <p className="text-xs font-mono text-slate-400 uppercase">Active Feed</p>
        </div>
        <div className="flex items-center gap-1.5">
          {isEmerg && <span className="text-[10px] font-mono text-red-400 bg-red-500/10 px-2 py-px rounded border border-red-500/30">EMERGENCY</span>}
          <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: meta.color }} />
          <span className="text-xs font-mono font-bold" style={{ color: meta.color }}>{meta.label}</span>
        </div>
      </div>
      <CCTVFeed road={activeRoad} videoUrl={videoUrls[activeRoad]} state={roads[activeRoad]} />
    </div>
  )
}

// ── Live Telemetry Dashboard ─────────────────────────────────
// Reads intersection_state from the polled /crossroad/state response
// (updated every 2 s). The backend SSE stream at /api/crossroad/telemetry/stream
// can be used instead for true real-time delivery.
const TELEMETRY_COLS = [
  { key: 'car',           label: 'Cars'  },
  { key: 'motorcycle',    label: 'Moto'  },
  { key: 'auto_rickshaw', label: 'Auto'  },
  { key: 'bus',           label: 'Bus'   },
]

function LiveDashboard({ intersectionState, signalMode, emergencyRoad }) {
  const lanes    = ROADS.map(r => intersectionState?.[r]).filter(Boolean)
  const hasData  = lanes.some(l => l.total > 0)
  const isEmerg  = signalMode === 'emergency'
  const maxTotal = Math.max(...lanes.map(l => l.total), 1)

  return (
    <div className="glass-card p-4 space-y-3">

      {/* Title bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-green-400 animate-pulse" />
          <p className="text-xs font-mono text-slate-400 uppercase tracking-wide">Live Telemetry Dashboard</p>
        </div>
        <div className="flex items-center gap-1.5">
          <motion.div
            animate={{ opacity: [1, 0.2, 1] }}
            transition={{ duration: 1.2, repeat: Infinity }}
            className="w-1.5 h-1.5 rounded-full bg-green-400"
          />
          <span className="text-[10px] font-mono text-green-400">LIVE</span>
        </div>
      </div>

      {/* Emergency alert banner */}
      <AnimatePresence>
        {isEmerg && emergencyRoad && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <motion.div
              animate={{ opacity: [1, 0.6, 1] }}
              transition={{ duration: 0.65, repeat: Infinity }}
              className="flex items-center gap-2 px-3 py-2 rounded-xl
                         bg-red-600/20 border border-red-500/50"
            >
              <Siren className="w-4 h-4 text-red-400 flex-shrink-0" />
              <p className="text-xs font-bold text-red-300">
                ⚠ ALERT: Ambulance in {emergencyRoad.charAt(0).toUpperCase() + emergencyRoad.slice(1)} Lane — Dashboard Warning Active
              </p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Vehicle-count table */}
      {hasData ? (
        <div className="overflow-hidden rounded-xl border border-white/8">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-white/5 border-b border-white/8">
                <th className="text-left px-3 py-2 font-mono text-slate-400 font-medium">Lane</th>
                {TELEMETRY_COLS.map(c => (
                  <th key={c.key} className="text-right px-2 py-2 font-mono text-slate-400 font-medium">{c.label}</th>
                ))}
                <th className="text-right px-3 py-2 font-mono text-slate-400 font-medium">Total</th>
              </tr>
            </thead>
            <tbody>
              {lanes.map((lane, i) => {
                const roadKey  = lane.lane.toLowerCase()
                const isAlert  = !!lane.alert
                const dotColor = ROAD_META[roadKey]?.color || '#3B82F6'
                return (
                  <motion.tr
                    key={lane.lane}
                    initial={{ opacity: 0, x: -4 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.04 }}
                    className={`border-b border-white/5 transition-colors duration-300 ${
                      isAlert ? 'bg-red-500/10' : i % 2 === 0 ? 'bg-white/[0.02]' : ''
                    }`}
                  >
                    <td className="px-3 py-2">
                      <div className="flex items-center gap-1.5">
                        {isAlert ? (
                          <motion.span
                            animate={{ opacity: [1, 0, 1] }}
                            transition={{ duration: 0.6, repeat: Infinity }}
                            className="w-2 h-2 rounded-full bg-red-400 flex-shrink-0"
                          />
                        ) : (
                          <span className="w-2 h-2 rounded-full flex-shrink-0"
                            style={{ backgroundColor: dotColor }} />
                        )}
                        <span className={`font-bold ${isAlert ? 'text-red-300' : 'text-white'}`}>
                          {lane.lane}
                        </span>
                      </div>
                    </td>
                    {TELEMETRY_COLS.map(c => (
                      <td key={c.key} className="text-right px-2 py-2 font-mono text-slate-300">
                        {lane.counts?.[c.key] ?? 0}
                      </td>
                    ))}
                    <td className="text-right px-3 py-2">
                      <span className={`font-bold font-mono ${isAlert ? 'text-red-300' : 'text-white'}`}>
                        {lane.total}
                      </span>
                    </td>
                  </motion.tr>
                )
              })}

              {/* Column totals */}
              <tr className="bg-white/5 border-t border-white/10">
                <td className="px-3 py-2 text-xs font-mono text-slate-500 font-bold">TOTAL</td>
                {TELEMETRY_COLS.map(c => (
                  <td key={c.key} className="text-right px-2 py-2 font-mono font-bold text-slate-300">
                    {lanes.reduce((s, l) => s + (l.counts?.[c.key] ?? 0), 0)}
                  </td>
                ))}
                <td className="text-right px-3 py-2 font-mono font-bold text-white">
                  {lanes.reduce((s, l) => s + l.total, 0)}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-center py-6 text-slate-600 font-mono text-xs">
          Waiting for YOLO sense-module telemetry…
        </div>
      )}

      {/* Per-lane load bars */}
      {hasData && (
        <div className="space-y-1.5">
          <p className="text-[10px] font-mono text-slate-500 uppercase">Load by lane</p>
          {lanes.map(lane => {
            const roadKey = lane.lane.toLowerCase()
            const pct     = (lane.total / maxTotal) * 100
            const color   = lane.alert ? '#EF4444' : ROAD_META[roadKey]?.color || '#3B82F6'
            return (
              <div key={lane.lane} className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-slate-400 w-12">{lane.lane}</span>
                <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                  <motion.div
                    animate={{ width: `${pct}%` }}
                    transition={{ duration: 0.5 }}
                    className="h-full rounded-full"
                    style={{ backgroundColor: color }}
                  />
                </div>
                <span className="text-[10px] font-mono text-white w-6 text-right">{lane.total}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Main Page ────────────────────────────────────────────────
export default function CrossroadMonitor() {
  const [crossroad, setCrossroad] = useState(null)
  const [models, setModels]       = useState({})
  const [location, setLocation]   = useState('')
  const [editLoc, setEditLoc]     = useState(false)
  const [videoUrls, setVideoUrls] = useState({})
  const pollRef  = useRef(null)
  const urlsRef  = useRef({})

  const fetchState = useCallback(async () => {
    try { const r = await api.get('/crossroad/state'); setCrossroad(r.data) }
    catch (err) { console.error(err) }
  }, [])

  useEffect(() => {
    api.get('/crossroad/models').then(r => setModels(r.data.models || {})).catch(console.error)
    fetchState()
    pollRef.current = setInterval(fetchState, 2000)
    return () => {
      clearInterval(pollRef.current)
      Object.values(urlsRef.current).forEach(u => URL.revokeObjectURL(u))
    }
  }, [fetchState])

  const handleUpload = async (road, file, modelKey) => {
    // Show video immediately via objectURL; backend processes in parallel
    const url = URL.createObjectURL(file)
    if (urlsRef.current[road]) URL.revokeObjectURL(urlsRef.current[road])
    urlsRef.current[road] = url
    setVideoUrls(prev => ({ ...prev, [road]: url }))

    try {
      const fd = new FormData()
      fd.append('video', file)
      fd.append('model', modelKey)
      if (location) fd.append('location', location)
      await api.post(`/crossroad/upload/${road}`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      })
      fetchState()
    } catch (err) {
      alert('Upload failed: ' + (err.response?.data?.error || err.message))
    }
  }

  const handleReset = async (road) => {
    if (urlsRef.current[road]) { URL.revokeObjectURL(urlsRef.current[road]); delete urlsRef.current[road] }
    setVideoUrls(prev => { const n = { ...prev }; delete n[road]; return n })
    try { await api.post(`/crossroad/reset/${road}`); fetchState() }
    catch (err) { console.error(err) }
  }

  const handleResetAll = async () => {
    if (!window.confirm('Reset all 4 roads?')) return
    Object.values(urlsRef.current).forEach(u => URL.revokeObjectURL(u))
    urlsRef.current = {}
    setVideoUrls({})
    try { await api.post('/crossroad/reset/all'); fetchState() }
    catch (err) { console.error(err) }
  }

  const handleAmbulance = async (road) => {
    try { await api.post(`/crossroad/ambulance/${road}`); fetchState() } catch (err) { console.error(err) }
  }
  const handleClearAmb = async (road) => {
    try { await api.post(`/crossroad/ambulance/${road}/clear`); fetchState() } catch (err) { console.error(err) }
  }
  const handleOverride = async (road, duration) => {
    try { await api.post(`/crossroad/signal/${road}/override`, { duration }); fetchState() } catch (err) { console.error(err) }
  }
  const handleSaveLocation = async () => {
    try { await api.post('/crossroad/settings', { location }); setEditLoc(false); fetchState() } catch (err) { console.error(err) }
  }

  const roads        = crossroad?.roads || {}
  const signalMode   = crossroad?.signal_mode || 'idle'
  const currentPhase = crossroad?.current_phase
  const phaseScores  = crossroad?.phase_scores
  const anyEmerg     = Object.values(roads).some(r => r?.ambulance_detected)
  const anyActive    = Object.values(roads).some(r => r?.status !== 'idle')
  const totalVehicles = Object.values(roads).reduce((a, r) => a + (r?.total_vehicles || 0), 0)
  const roadsUploaded = Object.values(roads).filter(r => r?.status !== 'idle').length
  const greenRoads    = ROADS.filter(r => roads[r]?.signal === 'green')

  return (
    <div className="p-4 space-y-5 max-w-[1400px] mx-auto">

      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="font-display font-bold text-2xl text-white">Indian Crossroad Controller</h1>
          <div className="flex items-center gap-2 mt-1">
            <MapPin className="w-3.5 h-3.5 text-slate-500" />
            {editLoc ? (
              <div className="flex items-center gap-2">
                <input autoFocus value={location} onChange={e => setLocation(e.target.value)}
                  placeholder="e.g. MG Road × Brigade Road, Bengaluru"
                  className="bg-slate-800 border border-white/10 rounded-lg px-2 py-1 text-xs text-white focus:outline-none w-64" />
                <button onClick={handleSaveLocation} className="text-xs text-green-400 hover:text-green-300">Save</button>
                <button onClick={() => setEditLoc(false)} className="text-xs text-slate-500">Cancel</button>
              </div>
            ) : (
              <button onClick={() => { setLocation(crossroad?.location || ''); setEditLoc(true) }}
                className="text-xs text-slate-400 hover:text-white transition-colors">
                {crossroad?.location || 'Click to set crossroad location'}
              </button>
            )}
          </div>
        </div>
        <button onClick={handleResetAll} className="btn-ghost py-2 px-4 text-sm flex items-center gap-2">
          <RotateCcw className="w-3.5 h-3.5" /> Reset All
        </button>
      </div>

      {/* Emergency banner */}
      <AnimatePresence>
        {anyEmerg && (
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
            className="flex items-center gap-3 p-3 rounded-xl bg-red-600/15 border border-red-500/30">
            <Siren className="w-5 h-5 text-red-400 animate-pulse flex-shrink-0" />
            <div>
              <p className="text-sm font-bold text-red-300">
                🚨 AMBULANCE PRIORITY — {crossroad?.emergency_road?.toUpperCase()} ROAD CLEARED
              </p>
              <p className="text-xs text-red-400/70 mt-0.5">90 seconds green · All phases suspended · 4-phase resumes after clearance</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main 3-column grid */}
      <div className="grid lg:grid-cols-3 gap-5">

        {/* Left — North + West */}
        <div className="space-y-4">
          {['north', 'west'].map(road => (
            <RoadPanel key={road} road={road} state={roads[road]} models={models}
              videoUrl={videoUrls[road]} onUpload={handleUpload} onAmbulance={handleAmbulance}
              onClearAmb={handleClearAmb} onOverride={handleOverride} onReset={handleReset} />
          ))}
        </div>

        {/* Center */}
        <div className="flex flex-col gap-5">

          {/* Active road CCTV feed */}
          <ActiveFeedPanel
            roads={roads} videoUrls={videoUrls}
            currentPhase={currentPhase} emergencyRoad={crossroad?.emergency_road} />

          {/* Crossroad diagram */}
          <div className="glass-card p-6 flex flex-col items-center gap-8">
            <CrossroadVisual roads={roads} currentPhase={currentPhase}
              emergencyRoad={crossroad?.emergency_road} signalMode={signalMode} />

            {greenRoads.length > 0 && !anyEmerg && (
              <div className="text-center w-full">
                <p className="text-xs text-slate-500 font-mono mb-1">ACTIVE GREEN</p>
                <div className="flex items-center gap-2 justify-center flex-wrap">
                  {greenRoads.map(r => (
                    <div key={r} className="flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-full bg-green-400 animate-pulse" />
                      <span className="font-display font-bold text-lg text-green-400 capitalize">{r}</span>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  {greenRoads.map(r => `${roads[r]?.total_vehicles || 0} vehicles`).join(' + ')} ·
                  PCU {greenRoads.map(r => roads[r]?.pcu_count || 0).reduce((a, b) => a + b, 0).toFixed(1)}
                  {currentPhase && ` · ${PHASES[currentPhase]?.movement}`}
                </p>
              </div>
            )}

            {anyEmerg && crossroad?.emergency_road && (
              <div className="text-center">
                <p className="text-xs text-slate-500 font-mono mb-1">EMERGENCY — ACTIVE GREEN</p>
                <div className="flex items-center gap-2 justify-center">
                  <div className="w-2.5 h-2.5 rounded-full bg-red-400 animate-pulse" />
                  <span className="font-display font-bold text-lg text-red-400 capitalize">
                    {crossroad.emergency_road} Road — 90s
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Phase indicator */}
          {anyActive && (
            <PhaseIndicator currentPhase={currentPhase} phaseScores={phaseScores} signalMode={signalMode} />
          )}

          {/* Summary stats */}
          {anyActive && (
            <div className="glass-card p-4 space-y-3">
              <p className="text-xs font-mono text-slate-500 uppercase">Crossroad Summary</p>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: 'Total Vehicles', value: totalVehicles, color: 'text-blue-400' },
                  { label: 'Roads Uploaded', value: `${roadsUploaded}/4`, color: 'text-green-400' },
                  { label: 'Signal Cycles',  value: crossroad?.cycle_count || 0, color: 'text-amber-400' },
                  { label: 'Mode', value: signalMode.toUpperCase(),
                    color: signalMode === 'emergency' ? 'text-red-400' : 'text-violet-400' },
                ].map(item => (
                  <div key={item.label} className="p-2.5 bg-white/3 rounded-xl border border-white/5 text-center">
                    <p className={`font-display font-bold text-lg ${item.color}`}>{item.value}</p>
                    <p className="text-xs text-slate-500">{item.label}</p>
                  </div>
                ))}
              </div>
              <div>
                <p className="text-xs font-mono text-slate-500 mb-2">ROAD PCU COMPARISON</p>
                {ROADS.map(road => {
                  const pcu    = roads[road]?.pcu_count || 0
                  const maxPcu = Math.max(...ROADS.map(r => roads[r]?.pcu_count || 0), 1)
                  const pct    = (pcu / maxPcu) * 100
                  const isGreen = roads[road]?.signal === 'green'
                  return (
                    <div key={road} className="flex items-center gap-2 mb-1.5">
                      <span className="text-xs font-mono text-slate-400 w-10 capitalize">{road}</span>
                      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                        <motion.div animate={{ width: `${pct}%` }} transition={{ duration: 0.6 }}
                          className="h-full rounded-full"
                          style={{ backgroundColor: isGreen ? '#22C55E' : ROAD_META[road].color }} />
                      </div>
                      <span className="text-xs font-mono text-white w-8 text-right">{pcu}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Live Telemetry Dashboard */}
          <LiveDashboard
            intersectionState={crossroad?.intersection_state}
            signalMode={signalMode}
            emergencyRoad={crossroad?.emergency_road}
          />

          {/* How-it-works (idle only) */}
          {!anyActive && (
            <div className="glass-card p-4 space-y-3">
              <p className="text-xs font-mono text-slate-500 uppercase">4-Phase Signal System</p>
              <div className="space-y-2 text-xs text-slate-400">
                {Object.entries(PHASES).map(([key, ph]) => (
                  <div key={key} className="flex gap-2.5">
                    <span className="font-mono font-black text-lg leading-none flex-shrink-0" style={{ color: ph.color }}>
                      {ph.arrows}
                    </span>
                    <div>
                      <p className="font-medium text-white">{ph.label} — {ph.movement}</p>
                      <p className="leading-relaxed mt-0.5">{ph.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="p-2.5 bg-white/3 rounded-xl border border-white/5 font-mono text-xs text-slate-500">
                <p className="text-slate-300 mb-1">ROI Lane Split — auto 65/35</p>
                <p>Left 65% of frame = straight lane · Right 35% = right-turn lane</p>
                <p className="mt-1">Phase with highest combined PCU score gets green automatically</p>
              </div>
            </div>
          )}
        </div>

        {/* Right — East + South */}
        <div className="space-y-4">
          {['east', 'south'].map(road => (
            <RoadPanel key={road} road={road} state={roads[road]} models={models}
              videoUrl={videoUrls[road]} onUpload={handleUpload} onAmbulance={handleAmbulance}
              onClearAmb={handleClearAmb} onOverride={handleOverride} onReset={handleReset} />
          ))}
        </div>
      </div>
    </div>
  )
}
