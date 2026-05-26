// frontend/src/pages/VideoMonitor.jsx
// Full video upload + YOLO detection page with all model versions

import React, { useState, useRef, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, Play, CheckCircle, AlertTriangle, Siren,
  Car, TruckIcon, Bus, Bike, Clock, BarChart3,
  Zap, RefreshCw, Trash2, ChevronDown, Film,
  Activity, Eye
} from 'lucide-react'
import api from '@/services/api'

// ── Constants ─────────────────────────────────────────────────
const DENSITY_COLOR = {
  low:    { text: 'text-green-400',  bg: 'bg-green-500/10',  border: 'border-green-500/25' },
  medium: { text: 'text-amber-400',  bg: 'bg-amber-500/10',  border: 'border-amber-500/25' },
  high:   { text: 'text-red-400',    bg: 'bg-red-500/10',    border: 'border-red-500/25'   },
}

const MODEL_GROUPS = [
  {
    group: 'YOLOv11 (Latest)',
    color: 'text-violet-400',
    models: ['yolov11n','yolov11s','yolov11m','yolov11l','yolov11x'],
  },
  {
    group: 'YOLOv10',
    color: 'text-blue-400',
    models: ['yolov10n','yolov10s','yolov10m','yolov10l'],
  },
  {
    group: 'YOLOv9',
    color: 'text-cyan-400',
    models: ['yolov9c','yolov9e'],
  },
  {
    group: 'YOLOv8 (Recommended)',
    color: 'text-crimson-400',
    models: ['yolov8n','yolov8s','yolov8m','yolov8l','yolov8x'],
  },
  {
    group: 'YOLOv5',
    color: 'text-amber-400',
    models: ['yolov5s','yolov5m','yolov5l'],
  },
  {
    group: 'YOLOv3 (Classic)',
    color: 'text-slate-400',
    models: ['yolov3'],
  },
]

const VEHICLE_ICONS = {
  car:        Car,
  truck:      TruckIcon,
  bus:        Bus,
  motorcycle: Bike,
  ambulance:  Siren,
}

const VEHICLE_COLORS = {
  car:        '#3B82F6',
  truck:      '#06B6D4',
  bus:        '#8B5CF6',
  motorcycle: '#F59E0B',
  ambulance:  '#EF4444',
}

// ── Sub-components ────────────────────────────────────────────

function ModelSelector({ models, selected, onSelect }) {
  const [open, setOpen] = useState(false)
  const selectedInfo = models[selected] || {}

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between gap-3 px-4 py-3
                   bg-slate-800/60 border border-white/10 rounded-xl
                   hover:border-white/20 transition-colors text-left"
      >
        <div>
          <p className="text-sm font-medium text-white">{selectedInfo.name || selected}</p>
          <p className="text-xs text-slate-400 mt-0.5">{selectedInfo.desc}</p>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <span className={`text-xs font-mono px-2 py-0.5 rounded-full border ${
            selectedInfo.speed === 'fast'
              ? 'text-green-400 border-green-500/25 bg-green-500/10'
              : selectedInfo.speed === 'medium'
              ? 'text-amber-400 border-amber-500/25 bg-amber-500/10'
              : 'text-red-400 border-red-500/25 bg-red-500/10'
          }`}>
            {selectedInfo.speed}
          </span>
          <span className="text-xs font-mono text-slate-400">
            mAP {selectedInfo.map}%
          </span>
          <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`} />
        </div>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="absolute top-full left-0 right-0 mt-2 z-50
                       bg-slate-800 border border-white/10 rounded-xl
                       overflow-hidden shadow-2xl max-h-96 overflow-y-auto"
          >
            {MODEL_GROUPS.map(grp => (
              <div key={grp.group}>
                <div className={`px-4 py-2 text-xs font-mono font-bold
                                 bg-white/3 border-b border-white/5 ${grp.color}`}>
                  {grp.group}
                </div>
                {grp.models.map(key => {
                  const info = models[key] || {}
                  return (
                    <button
                      key={key}
                      onClick={() => { onSelect(key); setOpen(false) }}
                      className={`w-full flex items-center justify-between gap-3
                                  px-4 py-2.5 text-left hover:bg-white/5
                                  transition-colors border-b border-white/3
                                  ${selected === key ? 'bg-crimson-600/15' : ''}`}
                    >
                      <div>
                        <p className={`text-sm font-medium ${selected === key ? 'text-crimson-400' : 'text-white'}`}>
                          {info.name}
                        </p>
                        <p className="text-xs text-slate-500">{info.desc}</p>
                      </div>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <span className={`text-xs font-mono ${
                          info.speed === 'fast' ? 'text-green-400'
                          : info.speed === 'medium' ? 'text-amber-400'
                          : 'text-red-400'
                        }`}>
                          {info.speed}
                        </span>
                        <span className="text-xs font-mono text-slate-400">
                          {info.map}%
                        </span>
                      </div>
                    </button>
                  )
                })}
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}


function ProgressBar({ value, label }) {
  return (
    <div>
      <div className="flex justify-between mb-1.5">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="text-xs font-mono text-white">{value}%</span>
      </div>
      <div className="h-2 bg-white/5 rounded-full overflow-hidden">
        <motion.div
          animate={{ width: `${value}%` }}
          transition={{ duration: 0.4 }}
          className="h-full rounded-full bg-gradient-to-r from-crimson-600 to-crimson-400"
        />
      </div>
    </div>
  )
}


function VehicleBreakdown({ totals }) {
  if (!totals || Object.keys(totals).length === 0) return null
  const total = Object.values(totals).reduce((a, b) => a + b, 0)
  return (
    <div className="space-y-2">
      {Object.entries(totals).map(([type, count]) => {
        const Icon = VEHICLE_ICONS[type] || Car
        const color = VEHICLE_COLORS[type] || '#94A3B8'
        const pct = total > 0 ? Math.round((count / total) * 100) : 0
        return (
          <div key={type}>
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <Icon className="w-3.5 h-3.5" style={{ color }} />
                <span className="text-xs text-slate-300 capitalize">{type}</span>
              </div>
              <span className="text-xs font-mono text-white">{count} ({pct}%)</span>
            </div>
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.8, delay: 0.1 }}
                className="h-full rounded-full"
                style={{ backgroundColor: color }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}


function FrameTimeline({ frames }) {
  if (!frames || frames.length === 0) return null
  const maxVeh = Math.max(...frames.map(f => f.total_vehicles), 1)

  return (
    <div>
      <p className="text-xs text-slate-500 font-mono mb-2">
        VEHICLE COUNT TIMELINE ({frames.length} frames)
      </p>
      <div className="flex items-end gap-0.5 h-16 bg-white/3 rounded-xl p-2">
        {frames.map((f, i) => {
          const h = Math.max(4, (f.total_vehicles / maxVeh) * 100)
          const color = f.density_class === 'high'
            ? '#EF4444'
            : f.density_class === 'medium'
            ? '#F59E0B'
            : '#22C55E'
          return (
            <div
              key={i}
              className="flex-1 rounded-sm transition-all duration-200
                         hover:opacity-80 cursor-default"
              style={{ height: `${h}%`, backgroundColor: color, minWidth: 2 }}
              title={`${f.timestamp_sec}s — ${f.total_vehicles} vehicles — ${f.density_class}`}
            />
          )
        })}
      </div>
      <div className="flex justify-between mt-1 text-xs text-slate-600 font-mono">
        <span>0s</span>
        <span>{frames[Math.floor(frames.length / 2)]?.timestamp_sec}s</span>
        <span>{frames[frames.length - 1]?.timestamp_sec}s</span>
      </div>
    </div>
  )
}


function JobCard({ job, onDelete }) {
  const isCompleted = job.status === 'completed'
  const isFailed    = job.status === 'failed'
  const density     = job.summary?.overall_density || 'low'
  const dc = DENSITY_COLOR[density] || DENSITY_COLOR.low

  return (
    <div className="glass-card p-4 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-white truncate">{job.filename}</p>
          <p className="text-xs text-slate-500 font-mono mt-0.5">
            {job.model_name} · {job.video_info?.duration_sec}s · {job.video_info?.resolution}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className={`text-xs font-mono px-2 py-0.5 rounded-full border ${
            isCompleted ? 'text-green-400 border-green-500/25 bg-green-500/10'
            : isFailed  ? 'text-red-400 border-red-500/25 bg-red-500/10'
            : 'text-amber-400 border-amber-500/25 bg-amber-500/10 animate-pulse'
          }`}>
            {job.status}
          </span>
          <button
            onClick={() => onDelete(job.job_id)}
            className="p-1 text-slate-600 hover:text-red-400 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {!isCompleted && !isFailed && (
        <ProgressBar value={job.progress || 0} label="Processing..." />
      )}

      {isCompleted && job.summary && (
        <div className="grid grid-cols-3 gap-2">
          <div className="text-center p-2 bg-white/3 rounded-xl border border-white/5">
            <p className="text-base font-bold text-white">
              {job.summary.avg_vehicles_per_frame}
            </p>
            <p className="text-xs text-slate-500">Avg/frame</p>
          </div>
          <div className={`text-center p-2 rounded-xl border ${dc.bg} ${dc.border}`}>
            <p className={`text-base font-bold ${dc.text}`}>
              {density.toUpperCase()}
            </p>
            <p className="text-xs text-slate-500">Density</p>
          </div>
          <div className="text-center p-2 bg-white/3 rounded-xl border border-white/5">
            <p className={`text-base font-bold ${
              job.summary.ambulance_detected ? 'text-red-400' : 'text-green-400'
            }`}>
              {job.summary.ambulance_detected ? '🚨 YES' : '✓ NO'}
            </p>
            <p className="text-xs text-slate-500">Ambulance</p>
          </div>
        </div>
      )}
    </div>
  )
}


// ── Main Page ─────────────────────────────────────────────────
export default function VideoMonitor() {
  const [models, setModels]           = useState({})
  const [selectedModel, setSelectedModel] = useState('yolov8s')
  const [dragOver, setDragOver]       = useState(false)
  const [uploading, setUploading]     = useState(false)
  const [currentJob, setCurrentJob]   = useState(null)
  const [result, setResult]           = useState(null)
  const [jobs, setJobs]               = useState([])
  const [videoPreview, setVideoPreview] = useState(null)
  const [videoFile, setVideoFile]     = useState(null)
  const [activeTab, setActiveTab]     = useState('upload')
  const fileRef = useRef(null)
  const pollRef = useRef(null)

  // Load models
  useEffect(() => {
    api.get('/video/models')
      .then(res => setModels(res.data.models || {}))
      .catch(console.error)
  }, [])

  // Load job history
  const loadJobs = useCallback(async () => {
    try {
      const res = await api.get('/video/jobs')
      setJobs(res.data.jobs || [])
    } catch (err) { console.error(err) }
  }, [])

  useEffect(() => { loadJobs() }, [loadJobs])

  // Poll job status
  const startPolling = useCallback((jobId) => {
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const res = await api.get(`/video/result/${jobId}`)
        const job = res.data
        setCurrentJob(job)
        if (job.status === 'completed' || job.status === 'failed') {
          clearInterval(pollRef.current)
          setResult(job)
          loadJobs()
          // Fetch full result with frames
          const full = await api.get(`/video/result/${jobId}`)
          setResult(full.data)
        }
      } catch (err) {
        console.error(err)
        clearInterval(pollRef.current)
      }
    }, 1500)
  }, [loadJobs])

  useEffect(() => () => clearInterval(pollRef.current), [])

  // Handle file select
  const handleFile = (file) => {
    if (!file) return
    const allowed = ['.mp4','.avi','.mov','.mkv','.webm','.wmv']
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!allowed.includes(ext)) {
      alert('Unsupported format. Use: MP4, AVI, MOV, MKV, WEBM, WMV')
      return
    }
    setVideoFile(file)
    setVideoPreview(URL.createObjectURL(file))
    setResult(null)
    setCurrentJob(null)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  // Upload and start processing
  const startProcessing = async () => {
    if (!videoFile) return
    setUploading(true)
    setResult(null)
    try {
      const formData = new FormData()
      formData.append('video', videoFile)
      formData.append('model', selectedModel)
      const res = await api.post('/video/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000,
      })
      const { job_id } = res.data
      setCurrentJob(res.data)
      setActiveTab('result')
      startPolling(job_id)
    } catch (err) {
      console.error('Upload error:', err)
      alert('Upload failed: ' + (err.response?.data?.error || err.message))
    } finally {
      setUploading(false)
    }
  }

  const deleteJob = async (jobId) => {
    try {
      await api.delete(`/video/jobs/${jobId}`)
      loadJobs()
      if (currentJob?.job_id === jobId) {
        setCurrentJob(null)
        setResult(null)
      }
    } catch (err) { console.error(err) }
  }

  const summary  = result?.summary || {}
  const frames   = result?.frame_results || []
  const density  = summary.overall_density || 'low'
  const dc       = DENSITY_COLOR[density] || DENSITY_COLOR.low

  return (
    <div className="p-4 space-y-4 max-w-[1400px] mx-auto">

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="font-display font-bold text-2xl text-white">
            Video Traffic Monitor
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Upload any traffic video · Choose YOLO version · Get AI detection results
          </p>
        </div>
        <div className="flex items-center gap-2">
          {['upload','result','history'].map(tab => (
            <button key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                activeTab === tab
                  ? 'bg-crimson-600 text-white'
                  : 'border border-white/10 text-slate-400 hover:text-white hover:border-white/20'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Upload Tab */}
      {activeTab === 'upload' && (
        <div className="grid lg:grid-cols-5 gap-5">

          {/* Left — Upload area + model selector */}
          <div className="lg:col-span-2 space-y-4">

            {/* Model selector */}
            <div className="glass-card p-5">
              <p className="text-xs font-mono text-slate-500 uppercase tracking-wider mb-3">
                Select YOLO Model
              </p>
              {Object.keys(models).length > 0 ? (
                <ModelSelector
                  models={models}
                  selected={selectedModel}
                  onSelect={setSelectedModel}
                />
              ) : (
                <div className="h-16 bg-white/3 rounded-xl animate-pulse" />
              )}
            </div>

            {/* Drop zone */}
            <div
              className={`glass-card p-6 border-2 border-dashed transition-all duration-200
                          flex flex-col items-center justify-center text-center cursor-pointer
                          min-h-[200px] ${
                dragOver
                  ? 'border-crimson-500/60 bg-crimson-500/5'
                  : 'border-white/15 hover:border-white/30'
              }`}
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
            >
              <input
                ref={fileRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={e => handleFile(e.target.files[0])}
              />
              <Film className={`w-10 h-10 mb-3 transition-colors ${
                dragOver ? 'text-crimson-400' : 'text-slate-600'
              }`} />
              <p className="text-sm font-medium text-white mb-1">
                {videoFile ? videoFile.name : 'Drop video here or click to browse'}
              </p>
              <p className="text-xs text-slate-500">
                MP4 · AVI · MOV · MKV · WEBM · WMV
              </p>
              {videoFile && (
                <p className="text-xs text-slate-400 mt-2 font-mono">
                  {(videoFile.size / 1024 / 1024).toFixed(1)} MB
                </p>
              )}
            </div>

            {/* Start button */}
            <button
              onClick={startProcessing}
              disabled={!videoFile || uploading}
              className="w-full btn-primary py-3 flex items-center justify-center
                         gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {uploading ? (
                <>
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white
                                   rounded-full animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Start {models[selectedModel]?.name || selectedModel} Detection
                </>
              )}
            </button>

            {/* Model comparison */}
            <div className="glass-card p-4">
              <p className="text-xs font-mono text-slate-500 uppercase mb-3">
                Speed vs Accuracy Guide
              </p>
              <div className="space-y-2 text-xs">
                {[
                  { label: 'Ultra Fast',    models: 'v8n, v10n, v11n', speed: 95, color: '#22C55E' },
                  { label: 'Fast',          models: 'v8s, v10s, v11s', speed: 80, color: '#86EFAC' },
                  { label: 'Balanced',      models: 'v8m, v9c, v11m',  speed: 60, color: '#F59E0B' },
                  { label: 'High Accuracy', models: 'v8l, v9e, v11l',  speed: 40, color: '#EF4444' },
                  { label: 'Max Accuracy',  models: 'v8x, v11x',       speed: 20, color: '#DC2626' },
                ].map(item => (
                  <div key={item.label}>
                    <div className="flex justify-between mb-0.5">
                      <span className="text-slate-300">{item.label}</span>
                      <span className="text-slate-500">{item.models}</span>
                    </div>
                    <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full rounded-full"
                        style={{ width: `${item.speed}%`, backgroundColor: item.color }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right — Video preview */}
          <div className="lg:col-span-3 space-y-4">
            <div className="glass-card overflow-hidden">
              <div className="px-4 py-3 border-b border-white/8 flex items-center gap-2">
                <Eye className="w-4 h-4 text-slate-400" />
                <span className="text-sm font-medium text-white">Video Preview</span>
              </div>
              <div className="relative bg-slate-900 aspect-video flex items-center
                              justify-center">
                {videoPreview ? (
                  <video
                    src={videoPreview}
                    controls
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-center">
                    <Film className="w-16 h-16 text-slate-700 mx-auto mb-3" />
                    <p className="text-slate-500 text-sm">
                      Select a video to preview
                    </p>
                    <p className="text-slate-600 text-xs mt-1">
                      Any traffic footage works — dashcam, CCTV, drone
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Tips */}
            <div className="glass-card p-5">
              <p className="text-xs font-mono text-slate-500 uppercase mb-3">
                Tips for Best Results
              </p>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { icon: '🎥', tip: 'Use MP4 format for fastest processing' },
                  { icon: '🚗', tip: 'Traffic intersection footage works best' },
                  { icon: '⚡', tip: 'Choose YOLOv8s for best speed/accuracy' },
                  { icon: '📏', tip: 'Shorter videos (< 2 min) process faster' },
                  { icon: '🌅', tip: 'Daylight footage gives better detection' },
                  { icon: '📹', tip: 'Dashcam or CCTV angle recommended' },
                ].map(item => (
                  <div key={item.tip}
                    className="flex items-start gap-2 p-2.5 bg-white/3 rounded-xl
                               border border-white/5">
                    <span className="text-base">{item.icon}</span>
                    <p className="text-xs text-slate-400 leading-relaxed">{item.tip}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Result Tab */}
      {activeTab === 'result' && (
        <div className="space-y-4">
          {/* Processing status */}
          {currentJob && currentJob.status !== 'completed' && currentJob.status !== 'failed' && (
            <div className="glass-card p-6 space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 border-2 border-crimson-500/30 border-t-crimson-500
                                rounded-full animate-spin flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-white">
                    Processing with {currentJob.model_name}
                  </p>
                  <p className="text-xs text-slate-400 mt-0.5">
                    {currentJob.filename} · {currentJob.video_info?.duration_sec}s video
                  </p>
                </div>
              </div>
              <ProgressBar
                value={currentJob.progress || 0}
                label={`Analyzing frames... (${currentJob.frames_done || 0} done)`}
              />
              <div className="grid grid-cols-3 gap-3 text-center">
                {[
                  ['Model',    currentJob.model_name],
                  ['Duration', `${currentJob.video_info?.duration_sec}s`],
                  ['FPS',      currentJob.video_info?.fps],
                ].map(([label, val]) => (
                  <div key={label} className="p-3 bg-white/3 rounded-xl border border-white/5">
                    <p className="text-sm font-mono font-bold text-white">{val}</p>
                    <p className="text-xs text-slate-500">{label}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Results display */}
          {result?.status === 'completed' && summary && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-4"
            >
              {/* Summary banner */}
              <div className={`glass-card p-5 border ${dc.border} ${dc.bg}`}>
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div>
                    <p className="text-xs font-mono text-slate-400 mb-1">
                      {result.model_name} · {result.filename}
                      {result.real_inference
                        ? ' · Real YOLOv8 inference'
                        : ' · Simulated detection'}
                    </p>
                    <h2 className={`font-display font-bold text-3xl ${dc.text}`}>
                      {density.toUpperCase()} TRAFFIC
                    </h2>
                    <p className="text-slate-400 text-sm mt-1">
                      Overall congestion level detected across {summary.total_frames_analyzed} frames
                    </p>
                  </div>
                  {summary.ambulance_detected && (
                    <div className="flex items-center gap-2 px-4 py-2 rounded-xl
                                    bg-red-600/15 border border-red-500/30 animate-pulse">
                      <Siren className="w-5 h-5 text-red-400" />
                      <div>
                        <p className="text-sm font-bold text-red-300">AMBULANCE DETECTED</p>
                        <p className="text-xs text-red-400">
                          at {summary.ambulance_timestamps?.slice(0,3).map(t => `${t}s`).join(', ')}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* KPI grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  {
                    label: 'Avg Vehicles/Frame',
                    value: summary.avg_vehicles_per_frame,
                    color: 'text-blue-400',
                    icon: Car,
                  },
                  {
                    label: 'Peak Vehicles',
                    value: summary.max_vehicles_in_frame,
                    color: 'text-red-400',
                    icon: Activity,
                  },
                  {
                    label: 'Frames Analyzed',
                    value: summary.total_frames_analyzed,
                    color: 'text-green-400',
                    icon: Film,
                  },
                  {
                    label: 'Avg Inference',
                    value: `${summary.avg_inference_ms}ms`,
                    color: 'text-amber-400',
                    icon: Zap,
                  },
                ].map(item => {
                  const Icon = item.icon
                  return (
                    <div key={item.label} className="stat-card">
                      <Icon className={`w-4 h-4 ${item.color}`} />
                      <div>
                        <p className={`font-display font-bold text-2xl ${item.color}`}>
                          {item.value}
                        </p>
                        <p className="text-xs text-slate-400">{item.label}</p>
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Charts row */}
              <div className="grid md:grid-cols-2 gap-4">
                {/* Timeline */}
                <div className="glass-card p-5">
                  <h3 className="font-display font-semibold text-sm text-white mb-4
                                 flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-slate-400" />
                    Vehicle Count Timeline
                  </h3>
                  <FrameTimeline frames={frames} />
                  <div className="flex gap-4 mt-3">
                    {[
                      { label: 'Low',    color: '#22C55E' },
                      { label: 'Medium', color: '#F59E0B' },
                      { label: 'High',   color: '#EF4444' },
                    ].map(item => (
                      <div key={item.label} className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-sm"
                          style={{ backgroundColor: item.color }} />
                        <span className="text-xs text-slate-400">{item.label}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Vehicle breakdown */}
                <div className="glass-card p-5">
                  <h3 className="font-display font-semibold text-sm text-white mb-4
                                 flex items-center gap-2">
                    <Car className="w-4 h-4 text-slate-400" />
                    Vehicle Type Breakdown
                  </h3>
                  <VehicleBreakdown totals={summary.vehicle_type_totals} />
                </div>
              </div>

              {/* Density distribution */}
              <div className="glass-card p-5">
                <h3 className="font-display font-semibold text-sm text-white mb-4">
                  Density Distribution
                </h3>
                <div className="grid grid-cols-3 gap-4">
                  {Object.entries(summary.density_breakdown || {}).map(([cls, count]) => {
                    const total = Object.values(summary.density_breakdown).reduce((a,b)=>a+b,0)
                    const pct = total > 0 ? Math.round((count/total)*100) : 0
                    const c = DENSITY_COLOR[cls] || DENSITY_COLOR.low
                    return (
                      <div key={cls} className={`p-4 rounded-xl border text-center ${c.bg} ${c.border}`}>
                        <p className={`font-display font-bold text-2xl ${c.text}`}>{pct}%</p>
                        <p className="text-xs text-slate-400 capitalize mt-1">{cls} traffic</p>
                        <p className="text-xs text-slate-500 font-mono">{count} frames</p>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Peak frame info */}
              {summary.peak_frame && (
                <div className="glass-card p-5 flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl bg-red-500/10 border border-red-500/25
                                  flex items-center justify-center flex-shrink-0">
                    <Activity className="w-5 h-5 text-red-400" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-white">Peak Congestion Frame</p>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Frame #{summary.peak_frame.frame_number} at {summary.peak_frame.timestamp_sec}s
                      — <span className="text-red-400 font-mono">
                        {summary.peak_frame.total_vehicles} vehicles
                      </span> detected
                    </p>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* Failed state */}
          {result?.status === 'failed' && (
            <div className="glass-card p-8 text-center">
              <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-3" />
              <p className="text-red-300 font-medium mb-1">Processing Failed</p>
              <p className="text-sm text-slate-400">{result.error}</p>
              <button onClick={() => setActiveTab('upload')}
                className="btn-primary mt-4 py-2 px-6 text-sm">
                Try Again
              </button>
            </div>
          )}

          {/* No result yet */}
          {!currentJob && !result && (
            <div className="glass-card p-12 text-center">
              <Film className="w-16 h-16 text-slate-700 mx-auto mb-4" />
              <p className="text-slate-400">No video processed yet</p>
              <button onClick={() => setActiveTab('upload')}
                className="btn-primary mt-4 py-2 px-6 text-sm flex items-center gap-2 mx-auto">
                <Upload className="w-4 h-4" /> Upload Video
              </button>
            </div>
          )}
        </div>
      )}

      {/* History Tab */}
      {activeTab === 'history' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-slate-400">{jobs.length} recent jobs</p>
            <button onClick={loadJobs}
              className="btn-ghost py-1.5 px-3 text-xs flex items-center gap-1.5">
              <RefreshCw className="w-3 h-3" /> Refresh
            </button>
          </div>
          {jobs.length === 0 ? (
            <div className="glass-card p-10 text-center">
              <Clock className="w-10 h-10 text-slate-700 mx-auto mb-3" />
              <p className="text-slate-400 text-sm">No jobs yet</p>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 gap-4">
              {jobs.map(job => (
                <JobCard key={job.job_id} job={job} onDelete={deleteJob} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
