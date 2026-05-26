// frontend/src/pages/CrossroadMonitor.jsx
// Single Indian crossroad — 4 road panels + visual signal map
// Real video upload per road, YOLO detection, auto signals, ambulance priority

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, RotateCcw, Siren, Zap, Car, Bus,
  TruckIcon, Bike, AlertTriangle, CheckCircle,
  ChevronUp, ChevronDown, ChevronLeft, ChevronRight,
  Film, MapPin, Settings, BarChart2, Clock
} from 'lucide-react'
import api from '@/services/api'

// ── Constants ─────────────────────────────────────────────────
const ROADS = ['north', 'south', 'east', 'west']

const ROAD_META = {
  north: { label: 'North Road',  icon: ChevronUp,    color: '#3B82F6', position: 'top'    },
  south: { label: 'South Road',  icon: ChevronDown,  color: '#22C55E', position: 'bottom' },
  east:  { label: 'East Road',   icon: ChevronRight, color: '#F59E0B', position: 'right'  },
  west:  { label: 'West Road',   icon: ChevronLeft,  color: '#8B5CF6', position: 'left'   },
}

const VEHICLE_META = {
  car:           { icon: Car,       color: '#3B82F6', label: 'Cars'         },
  motorcycle:    { icon: Bike,      color: '#F59E0B', label: 'Motorcycles'  },
  auto_rickshaw: { icon: Car,       color: '#F97316', label: 'Auto Rickshaw'},
  bus:           { icon: Bus,       color: '#8B5CF6', label: 'Buses'        },
  truck:         { icon: TruckIcon, color: '#06B6D4', label: 'Trucks'       },
  bicycle:       { icon: Bike,      color: '#6EE7B7', label: 'Bicycles'     },
  pedestrian:    { icon: Car,       color: '#94A3B8', label: 'Pedestrians'  },
  ambulance:     { icon: Siren,     color: '#EF4444', label: 'Ambulance'    },
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

const ALLOWED = ['.mp4','.avi','.mov','.mkv','.webm','.wmv']

// ── Traffic Light Component ───────────────────────────────────
function TrafficLight({ phase, size = 'md' }) {
  const dim = size === 'lg' ? 'w-8 h-8' : size === 'md' ? 'w-5 h-5' : 'w-3 h-3'
  return (
    <div className="flex flex-col gap-1.5 bg-slate-900 border border-white/15
                    rounded-xl p-2 items-center shadow-xl">
      {['red','yellow','green'].map(c => (
        <div key={c} className={`${dim} rounded-full transition-all duration-500`}
          style={{
            backgroundColor: phase === c ? SIGNAL_GLOW[c].bg : 'rgba(255,255,255,0.05)',
            boxShadow: phase === c ? SIGNAL_GLOW[c].glow : 'none',
          }}
        />
      ))}
    </div>
  )
}

// ── Central Crossroad Visual ──────────────────────────────────
function CrossroadVisual({ roads, activeRoad, emergencyRoad, signalMode }) {
  return (
    <div className="relative w-full aspect-square max-w-[320px] mx-auto select-none">

      {/* Road strips */}
      {/* Horizontal road (East-West) */}
      <div className="absolute inset-y-[38%] inset-x-0 bg-slate-700/60
                      border-y border-white/10" />
      {/* Vertical road (North-South) */}
      <div className="absolute inset-x-[38%] inset-y-0 bg-slate-700/60
                      border-x border-white/10" />

      {/* Center box */}
      <div className="absolute inset-[38%] bg-slate-600/80 border border-white/15
                      flex items-center justify-center z-10">
        <div className="w-3 h-3 rounded-full bg-white/20" />
      </div>

      {/* Lane markings */}
      <div className="absolute left-1/2 top-0 bottom-[38%] w-px
                      border-l-2 border-dashed border-white/20 -translate-x-1/2" />
      <div className="absolute left-1/2 top-[62%] bottom-0 w-px
                      border-l-2 border-dashed border-white/20 -translate-x-1/2" />
      <div className="absolute top-1/2 left-0 right-[38%] h-px
                      border-t-2 border-dashed border-white/20 -translate-y-1/2" />
      <div className="absolute top-1/2 left-[62%] right-0 h-px
                      border-t-2 border-dashed border-white/20 -translate-y-1/2" />

      {/* Per-road signal + label */}
      {ROADS.map(road => {
        const meta    = ROAD_META[road]
        const state   = roads[road]
        const phase   = state?.signal || 'red'
        const isActive = road === activeRoad
        const isEmerg  = road === emergencyRoad
        const vc       = state?.total_vehicles || 0

        const posClass = {
          top:    'top-1 left-1/2 -translate-x-1/2 flex-col items-center',
          bottom: 'bottom-1 left-1/2 -translate-x-1/2 flex-col-reverse items-center',
          left:   'left-1 top-1/2 -translate-y-1/2 flex-row items-center',
          right:  'right-1 top-1/2 -translate-y-1/2 flex-row-reverse items-center',
        }[meta.position]

        return (
          <div key={road}
            className={`absolute flex gap-2 z-20 ${posClass}`}>
            <TrafficLight phase={phase} size="sm" />
            <div className="text-center">
              <p className="text-xs font-mono font-bold"
                style={{ color: isEmerg ? '#EF4444' : isActive ? '#22C55E' : '#64748b' }}>
                {road.toUpperCase()}
              </p>
              {vc > 0 && (
                <p className="text-xs font-bold text-white">{vc}</p>
              )}
            </div>
          </div>
        )
      })}

      {/* Emergency pulse overlay */}
      {emergencyRoad && (
        <motion.div
          animate={{ opacity: [0.2, 0.5, 0.2] }}
          transition={{ duration: 0.8, repeat: Infinity }}
          className="absolute inset-0 rounded-full border-4 border-red-500/40 z-30"
        />
      )}

      {/* Signal mode badge */}
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

// ── Frame Timeline ────────────────────────────────────────────
function Timeline({ frames, color }) {
  if (!frames?.length) return null
  const max = Math.max(...frames.map(f => f.total_vehicles), 1)
  return (
    <div className="flex items-end gap-px h-8 rounded-lg overflow-hidden bg-white/3 p-1">
      {frames.map((f, i) => {
        const h = Math.max(10, (f.total_vehicles / max) * 100)
        const c = f.density_class === 'high' ? '#EF4444'
          : f.density_class === 'medium' ? '#F59E0B' : '#22C55E'
        return (
          <div key={i} className="flex-1 rounded-sm min-w-0.5 transition-all"
            style={{ height: `${h}%`, backgroundColor: c }}
            title={`${f.timestamp_sec}s — ${f.total_vehicles} vehicles`} />
        )
      })}
    </div>
  )
}

// ── Road Panel ────────────────────────────────────────────────
function RoadPanel({ road, state, models, onUpload, onAmbulance, onClearAmb, onOverride, onReset }) {
  const [model, setModel]         = useState('yolov8s')
  const [drag, setDrag]           = useState(false)
  const [showOverride, setOver]   = useState(false)
  const [overDur, setOverDur]     = useState(30)
  const [frames, setFrames]       = useState([])
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
      api.get(`/crossroad/frames/${road}`)
        .then(r => setFrames(r.data.frames || []))
        .catch(() => {})
    }
  }, [status, road])

  const handleFile = (file) => {
    if (!file) return
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!ALLOWED.includes(ext)) {
      alert('Unsupported format. Use: ' + ALLOWED.join(', '))
      return
    }
    onUpload(road, file, model)
  }

  return (
    <div className={`glass-card flex flex-col gap-3 p-4 border transition-all duration-500 ${
      isEmerg      ? 'border-red-500/50 bg-red-500/5'
      : phase === 'green' ? 'border-green-500/30 bg-green-500/3'
      : 'border-white/8'
    }`}>

      {/* Road header */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl flex items-center justify-center border"
            style={{
              borderColor: meta.color + '40',
              backgroundColor: meta.color + '15',
              color: meta.color,
            }}>
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
            <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded-full border
                             ${ds.text} ${ds.bg} ${ds.border}`}>
              {density.toUpperCase()}
            </span>
          )}
        </div>
      </div>

      {/* Ambulance alert */}
      <AnimatePresence>
        {isEmerg && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="flex items-center justify-between p-2.5 rounded-xl
                       bg-red-600/15 border border-red-500/30"
          >
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
              className="text-xs px-2 py-1 border border-white/15 rounded-lg
                         text-slate-300 hover:text-white transition-colors flex-shrink-0">
              Clear
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* IDLE — show upload */}
      {status === 'idle' && (
        <div className="space-y-3">
          {/* Model selector */}
          <div>
            <p className="text-xs font-mono text-slate-500 mb-1.5">YOLO MODEL</p>
            <select
              value={model}
              onChange={e => setModel(e.target.value)}
              className="w-full bg-slate-800 border border-white/10 rounded-xl px-3 py-2
                         text-sm text-white focus:outline-none focus:border-crimson-500/40"
            >
              <optgroup label="── YOLOv11 (Latest)">
                {['yolov11n','yolov11s','yolov11m'].map(k => (
                  <option key={k} value={k}>
                    {models[k]?.name || k} — {models[k]?.speed} · {models[k]?.map}% mAP
                  </option>
                ))}
              </optgroup>
              <optgroup label="── YOLOv8 ✅ Recommended">
                {['yolov8n','yolov8s','yolov8m','yolov8l'].map(k => (
                  <option key={k} value={k}>
                    {models[k]?.name || k} — {models[k]?.speed} · {models[k]?.map}% mAP
                    {models[k]?.rec ? ' ★' : ''}
                  </option>
                ))}
              </optgroup>
              <optgroup label="── YOLOv9 / v10">
                {['yolov9c','yolov10s'].map(k => (
                  <option key={k} value={k}>
                    {models[k]?.name || k} — {models[k]?.speed}
                  </option>
                ))}
              </optgroup>
            </select>
          </div>

          {/* Drop zone */}
          <div
            className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer
                        transition-all duration-200 ${
              drag
                ? 'border-crimson-500/60 bg-crimson-500/8'
                : 'border-white/15 hover:border-white/30 hover:bg-white/3'
            }`}
            onDragOver={e => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={e => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]) }}
            onClick={() => fileRef.current?.click()}
          >
            <input ref={fileRef} type="file" accept="video/*" className="hidden"
              onChange={e => handleFile(e.target.files[0])} />
            <Film className={`w-8 h-8 mx-auto mb-2 transition-colors ${
              drag ? 'text-crimson-400' : 'text-slate-600'
            }`} />
            <p className="text-sm font-medium text-white mb-1">
              Drop {meta.label} video
            </p>
            <p className="text-xs text-slate-500">MP4 · AVI · MOV · MKV</p>
          </div>

          <p className="text-xs text-slate-600 text-center leading-relaxed">
            Upload CCTV / dashcam video from the {meta.label.toLowerCase()} direction.
            AI will detect all Indian vehicles and update signals.
          </p>
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
              <motion.div
                animate={{ width: `${state?.progress || 0}%` }}
                transition={{ duration: 0.4 }}
                className="h-full rounded-full bg-gradient-to-r from-crimson-600 to-crimson-400"
              />
            </div>
          </div>
          <p className="text-xs text-slate-500 text-center">
            {state?.filename} · {state?.video_info?.duration_sec}s ·
            {models[state?.model_used]?.name}
          </p>
          {state?.current_frame && (
            <div className="p-2.5 bg-white/3 rounded-xl border border-white/5">
              <p className="text-xs text-slate-500 font-mono mb-1.5">LATEST FRAME</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(state.current_frame.vehicle_counts || {}).map(([cls, cnt]) => (
                  <span key={cls} className="text-xs font-mono text-white">
                    <span className="text-slate-400">{cls}:</span> {cnt}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* FAILED */}
      {status === 'failed' && (
        <div className="flex flex-col items-center gap-2 py-3">
          <AlertTriangle className="w-8 h-8 text-red-400" />
          <p className="text-sm text-red-300">Processing failed</p>
          <button onClick={() => onReset(road)}
            className="btn-ghost text-xs py-1.5 px-4">
            Reset &amp; Try Again
          </button>
        </div>
      )}

      {/* COMPLETED + live frame preview */}
      {(status === 'completed' || (status === 'processing' && state?.current_frame)) && (
        <div className="space-y-3">
          {/* Vehicle type breakdown */}
          <div>
            <p className="text-xs text-slate-500 font-mono mb-2">DETECTED VEHICLES (avg/frame)</p>
            <div className="grid grid-cols-2 gap-1.5">
              {Object.entries(state?.vehicle_counts || {}).map(([cls, cnt]) => {
                const vm = VEHICLE_META[cls] || VEHICLE_META.car
                const VIcon = vm.icon
                return (
                  <div key={cls}
                    className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg
                               bg-white/3 border border-white/5">
                    <VIcon className="w-3.5 h-3.5 flex-shrink-0"
                      style={{ color: vm.color }} />
                    <span className="text-xs text-slate-300 capitalize truncate flex-1">
                      {cls.replace('_',' ')}
                    </span>
                    <span className="text-xs font-bold text-white">{cnt}</span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* PCU + green time */}
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="p-2 bg-white/3 rounded-xl border border-white/5">
              <p className="text-base font-bold text-white">{state?.total_vehicles || 0}</p>
              <p className="text-xs text-slate-500">Avg vehicles</p>
            </div>
            <div className="p-2 bg-white/3 rounded-xl border border-white/5">
              <p className="text-base font-bold text-amber-400">{state?.pcu_count || 0}</p>
              <p className="text-xs text-slate-500">PCU score</p>
            </div>
            <div className={`p-2 rounded-xl border ${
              phase === 'green'
                ? 'bg-green-500/10 border-green-500/25'
                : 'bg-white/3 border-white/5'
            }`}>
              <p className={`text-base font-bold ${
                phase === 'green' ? 'text-green-400' : 'text-slate-400'
              }`}>{state?.green_duration || 0}s</p>
              <p className="text-xs text-slate-500">Green time</p>
            </div>
          </div>

          {/* Timeline */}
          {frames.length > 0 && (
            <div>
              <p className="text-xs text-slate-500 font-mono mb-1">
                TRAFFIC TIMELINE — {frames.length} frames
              </p>
              <Timeline frames={frames} color={meta.color} />
            </div>
          )}

          {/* Ambulance timestamps */}
          {state?.ambulance_timestamps?.length > 0 && (
            <div className="p-2 bg-red-500/8 rounded-xl border border-red-500/20">
              <p className="text-xs font-mono text-red-400 mb-1">🚨 AMBULANCE DETECTED</p>
              <p className="text-xs text-slate-400">
                At: {state.ambulance_timestamps.slice(0,5).map(t=>`${t}s`).join(' · ')}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Signal info */}
      {phase === 'green' && state?.green_duration > 0 && (
        <div className="flex items-center gap-2 p-2 bg-green-500/8 rounded-xl
                        border border-green-500/20">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <p className="text-xs text-green-300 font-mono">
            GREEN — {state.green_duration}s · {state.total_vehicles} vehicles waiting
          </p>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2 flex-wrap border-t border-white/5 pt-3">
        {status !== 'idle' && status !== 'processing' && (
          <button onClick={() => onReset(road)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs
                       border border-white/10 text-slate-400 hover:text-white
                       hover:border-white/25 transition-colors">
            <RotateCcw className="w-3 h-3" /> New Video
          </button>
        )}
        {!isEmerg ? (
          <button onClick={() => onAmbulance(road)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs
                       bg-red-600/10 border border-red-500/25 text-red-400
                       hover:bg-red-600/20 transition-colors">
            <Siren className="w-3 h-3" /> Ambulance
          </button>
        ) : (
          <button onClick={() => onClearAmb(road)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs
                       bg-green-500/10 border border-green-500/25 text-green-400
                       hover:bg-green-500/20 transition-colors">
            <CheckCircle className="w-3 h-3" /> Clear
          </button>
        )}
        <button onClick={() => setOver(!showOverride)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs
                     border border-white/10 text-slate-400 hover:text-white
                     hover:border-white/25 transition-colors ml-auto">
          Override
        </button>
      </div>

      {/* Manual override */}
      <AnimatePresence>
        {showOverride && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden border-t border-white/8 pt-3 space-y-2"
          >
            <p className="text-xs text-slate-500 font-mono">MANUAL OVERRIDE</p>
            <div className="flex items-center gap-2">
              <input
                type="number" min={5} max={120} value={overDur}
                onChange={e => setOverDur(+e.target.value)}
                className="w-20 bg-slate-800 border border-white/10 rounded-lg
                           px-2 py-1.5 text-xs text-white font-mono
                           focus:outline-none text-center"
              />
              <span className="text-xs text-slate-500">seconds green</span>
              <button
                onClick={() => { onOverride(road, overDur); setOver(false) }}
                className="flex-1 py-1.5 bg-crimson-600 hover:bg-crimson-700
                           text-white text-xs rounded-lg font-medium transition-colors">
                Apply
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────
export default function CrossroadMonitor() {
  const [crossroad, setCrossroad] = useState(null)
  const [models, setModels]       = useState({})
  const [location, setLocation]   = useState('')
  const [editLoc, setEditLoc]     = useState(false)
  const pollRef = useRef(null)

  const fetchState = useCallback(async () => {
    try {
      const r = await api.get('/crossroad/state')
      setCrossroad(r.data)
    } catch (err) { console.error(err) }
  }, [])

  useEffect(() => {
    api.get('/crossroad/models').then(r => setModels(r.data.models || {})).catch(console.error)
    fetchState()
    pollRef.current = setInterval(fetchState, 2000)
    return () => clearInterval(pollRef.current)
  }, [fetchState])

  const handleUpload = async (road, file, modelKey) => {
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

  const handleAmbulance = async (road) => {
    try { await api.post(`/crossroad/ambulance/${road}`); fetchState() }
    catch (err) { console.error(err) }
  }

  const handleClearAmb = async (road) => {
    try { await api.post(`/crossroad/ambulance/${road}/clear`); fetchState() }
    catch (err) { console.error(err) }
  }

  const handleOverride = async (road, duration) => {
    try { await api.post(`/crossroad/signal/${road}/override`, { duration }); fetchState() }
    catch (err) { console.error(err) }
  }

  const handleReset = async (road) => {
    try { await api.post(`/crossroad/reset/${road}`); fetchState() }
    catch (err) { console.error(err) }
  }

  const handleResetAll = async () => {
    if (!window.confirm('Reset all 4 roads?')) return
    try { await api.post('/crossroad/reset/all'); fetchState() }
    catch (err) { console.error(err) }
  }

  const handleSaveLocation = async () => {
    try {
      await api.post('/crossroad/settings', { location })
      setEditLoc(false)
      fetchState()
    } catch (err) { console.error(err) }
  }

  const roads      = crossroad?.roads || {}
  const activeRoad = crossroad?.active_road
  const signalMode = crossroad?.signal_mode || 'idle'
  const anyEmerg   = Object.values(roads).some(r => r?.ambulance_detected)
  const anyActive  = Object.values(roads).some(r => r?.status !== 'idle')

  // Summary stats
  const totalVehicles = Object.values(roads).reduce((a, r) => a + (r?.total_vehicles || 0), 0)
  const roadsUploaded = Object.values(roads).filter(r => r?.status !== 'idle').length

  return (
    <div className="p-4 space-y-5 max-w-[1400px] mx-auto">

      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="font-display font-bold text-2xl text-white">
            Indian Crossroad Controller
          </h1>
          {/* Location */}
          <div className="flex items-center gap-2 mt-1">
            <MapPin className="w-3.5 h-3.5 text-slate-500" />
            {editLoc ? (
              <div className="flex items-center gap-2">
                <input
                  autoFocus
                  value={location}
                  onChange={e => setLocation(e.target.value)}
                  placeholder="e.g. MG Road × Brigade Road, Bengaluru"
                  className="bg-slate-800 border border-white/10 rounded-lg px-2 py-1
                             text-xs text-white focus:outline-none w-64"
                />
                <button onClick={handleSaveLocation}
                  className="text-xs text-green-400 hover:text-green-300">Save</button>
                <button onClick={() => setEditLoc(false)}
                  className="text-xs text-slate-500">Cancel</button>
              </div>
            ) : (
              <button onClick={() => { setLocation(crossroad?.location || ''); setEditLoc(true) }}
                className="text-xs text-slate-400 hover:text-white transition-colors">
                {crossroad?.location || 'Click to set crossroad location'}
              </button>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleResetAll}
            className="btn-ghost py-2 px-4 text-sm flex items-center gap-2">
            <RotateCcw className="w-3.5 h-3.5" /> Reset All
          </button>
        </div>
      </div>

      {/* Emergency banner */}
      <AnimatePresence>
        {anyEmerg && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex items-center gap-3 p-3 rounded-xl
                       bg-red-600/15 border border-red-500/30"
          >
            <Siren className="w-5 h-5 text-red-400 animate-pulse flex-shrink-0" />
            <div>
              <p className="text-sm font-bold text-red-300">
                🚨 AMBULANCE PRIORITY — {crossroad?.emergency_road?.toUpperCase()} ROAD CLEARED
              </p>
              <p className="text-xs text-red-400/70 mt-0.5">
                90 seconds green · All other roads 0s · System in emergency mode
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main layout — crossroad visual + 4 panels */}
      <div className="grid lg:grid-cols-3 gap-5">

        {/* Left col — North + West */}
        <div className="space-y-4">
          <RoadPanel road="north" state={roads.north} models={models}
            onUpload={handleUpload} onAmbulance={handleAmbulance}
            onClearAmb={handleClearAmb} onOverride={handleOverride}
            onReset={handleReset} />
          <RoadPanel road="west" state={roads.west} models={models}
            onUpload={handleUpload} onAmbulance={handleAmbulance}
            onClearAmb={handleClearAmb} onOverride={handleOverride}
            onReset={handleReset} />
        </div>

        {/* Center — crossroad visual + stats */}
        <div className="flex flex-col gap-5">
          {/* Crossroad diagram */}
          <div className="glass-card p-6 flex flex-col items-center gap-8">
            <CrossroadVisual
              roads={roads}
              activeRoad={activeRoad}
              emergencyRoad={crossroad?.emergency_road}
              signalMode={signalMode}
            />

            {/* Active road indicator */}
            {activeRoad && (
              <div className="text-center">
                <p className="text-xs text-slate-500 font-mono mb-1">ACTIVE GREEN</p>
                <div className="flex items-center gap-2 justify-center">
                  <div className="w-2.5 h-2.5 rounded-full bg-green-400 animate-pulse" />
                  <span className="font-display font-bold text-lg text-green-400">
                    {ROAD_META[activeRoad]?.label}
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  {roads[activeRoad]?.total_vehicles || 0} vehicles ·
                  PCU {roads[activeRoad]?.pcu_count || 0} ·
                  {roads[activeRoad]?.green_duration || 0}s green
                </p>
              </div>
            )}
          </div>

          {/* Summary stats */}
          {anyActive && (
            <div className="glass-card p-4 space-y-3">
              <p className="text-xs font-mono text-slate-500 uppercase">Crossroad Summary</p>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: 'Total Vehicles',   value: totalVehicles,      color: 'text-blue-400' },
                  { label: 'Roads Uploaded',   value: `${roadsUploaded}/4`, color: 'text-green-400' },
                  { label: 'Signal Cycles',    value: crossroad?.cycle_count || 0, color: 'text-amber-400' },
                  { label: 'Mode',             value: signalMode.toUpperCase(), color: signalMode === 'emergency' ? 'text-red-400' : 'text-violet-400' },
                ].map(item => (
                  <div key={item.label} className="p-2.5 bg-white/3 rounded-xl border border-white/5 text-center">
                    <p className={`font-display font-bold text-lg ${item.color}`}>{item.value}</p>
                    <p className="text-xs text-slate-500">{item.label}</p>
                  </div>
                ))}
              </div>

              {/* Per-road PCU bar chart */}
              <div>
                <p className="text-xs font-mono text-slate-500 mb-2">ROAD PCU COMPARISON</p>
                {ROADS.map(road => {
                  const pcu   = roads[road]?.pcu_count || 0
                  const maxPcu = Math.max(...ROADS.map(r => roads[r]?.pcu_count || 0), 1)
                  const pct   = maxPcu > 0 ? (pcu / maxPcu) * 100 : 0
                  const isGreen = road === activeRoad
                  return (
                    <div key={road} className="flex items-center gap-2 mb-1.5">
                      <span className="text-xs font-mono text-slate-400 w-10 capitalize">{road}</span>
                      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                        <motion.div
                          animate={{ width: `${pct}%` }}
                          transition={{ duration: 0.6 }}
                          className="h-full rounded-full"
                          style={{ backgroundColor: isGreen ? '#22C55E' : ROAD_META[road].color }}
                        />
                      </div>
                      <span className="text-xs font-mono text-white w-8 text-right">{pcu}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* How it works — only when idle */}
          {!anyActive && (
            <div className="glass-card p-4 space-y-3">
              <p className="text-xs font-mono text-slate-500 uppercase">How To Use</p>
              <div className="space-y-2 text-xs text-slate-400">
                {[
                  { n:'1', t:'Set Location', d:'Click the location text above to name your crossroad.' },
                  { n:'2', t:'Upload Videos', d:'Drop a CCTV/dashcam video on each road panel. You can upload 1, 2, 3, or all 4 roads.' },
                  { n:'3', t:'AI Detection', d:'YOLO detects cars, bikes, autos, buses, trucks, pedestrians and ambulances.' },
                  { n:'4', t:'Auto Signals', d:'The road with most traffic (highest PCU score) gets the green signal automatically.' },
                  { n:'5', t:'Ambulance', d:'If ambulance detected in video or triggered manually → 90s green, all others 0s.' },
                ].map(item => (
                  <div key={item.n} className="flex gap-2.5">
                    <span className="font-mono text-crimson-500 font-bold flex-shrink-0">{item.n}.</span>
                    <div>
                      <p className="font-medium text-white">{item.t}</p>
                      <p className="leading-relaxed mt-0.5">{item.d}</p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="p-2.5 bg-white/3 rounded-xl border border-white/5 font-mono text-xs text-slate-500">
                <p className="text-slate-300 mb-1">PCU (Passenger Car Unit) — Indian Roads Congress</p>
                <p>Car=1.0 · Auto=0.8 · Bike=0.5 · Bus=2.5 · Truck=2.0 · Ambulance=10.0 (priority)</p>
              </div>
            </div>
          )}
        </div>

        {/* Right col — East + South */}
        <div className="space-y-4">
          <RoadPanel road="east" state={roads.east} models={models}
            onUpload={handleUpload} onAmbulance={handleAmbulance}
            onClearAmb={handleClearAmb} onOverride={handleOverride}
            onReset={handleReset} />
          <RoadPanel road="south" state={roads.south} models={models}
            onUpload={handleUpload} onAmbulance={handleAmbulance}
            onClearAmb={handleClearAmb} onOverride={handleOverride}
            onReset={handleReset} />
        </div>
      </div>
    </div>
  )
}
