// frontend/src/pages/LiveMonitor.jsx
import React, { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Square, Upload, Zap, Car, Bus, TruckIcon, Bike, Siren, Radio } from 'lucide-react'
import { detectionAPI, trafficAPI } from '@/services/api'

const CLASS_COLORS = {
  car: '#3B82F6', motorcycle: '#F59E0B', bus: '#8B5CF6',
  truck: '#06B6D4', ambulance: '#EF4444'
}

function DetectionBox({ det, frameW, frameH }) {
  const [x1, y1, x2, y2] = det.bbox
  const scaleX = 100 / (frameW || 640)
  const scaleY = 100 / (frameH || 480)
  const color = CLASS_COLORS[det.class] || '#fff'
  return (
    <div
      className="absolute border-2 rounded transition-all duration-100"
      style={{
        left: `${x1 * scaleX}%`, top: `${y1 * scaleY}%`,
        width: `${(x2 - x1) * scaleX}%`, height: `${(y2 - y1) * scaleY}%`,
        borderColor: color,
      }}
    >
      <span className="absolute -top-5 left-0 text-xs font-mono px-1 py-0.5 rounded whitespace-nowrap"
        style={{ backgroundColor: color, color: '#fff' }}>
        {det.track_id > 0 ? `#${det.track_id} ` : ''}{det.class} {(det.confidence * 100).toFixed(0)}%
      </span>
    </div>
  )
}

function LiveStat({ label, value, color = 'text-white' }) {
  return (
    <div className="text-center">
      <p className={`font-display font-bold text-xl ${color}`}>{value}</p>
      <p className="text-xs text-slate-500 mt-0.5">{label}</p>
    </div>
  )
}

export default function LiveMonitor() {
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [uploadedImage, setUploadedImage] = useState(null)
  const [mode, setMode] = useState('simulate') // 'simulate' | 'upload'
  const intervalRef = useRef(null)
  const fileRef = useRef(null)

  const runDetection = useCallback(async () => {
    try {
      const { data } = await detectionAPI.detectVehiclesBase64('')
      setResult(data)
      setHistory(prev => [
        { ...data, ts: new Date().toLocaleTimeString() },
        ...prev.slice(0, 49)
      ])
      // Push to backend DB
      if (data.total_vehicles > 0) {
        trafficAPI.addRecord({
          intersection_id: 1,
          lane: 'north',
          vehicle_type: Object.keys(data.vehicle_counts || {})[0] || 'car',
          vehicle_count: data.total_vehicles,
          density_class: data.density_class,
          congestion_score: data.density_class === 'high' ? 0.85 : data.density_class === 'medium' ? 0.50 : 0.15,
          ambulance_detected: data.ambulance_detected,
          confidence: 0.92,
        }).catch(() => {})
      }
    } catch (err) {
      console.error('Detection error:', err)
    }
  }, [])

  const start = () => {
    setRunning(true)
    runDetection()
    intervalRef.current = setInterval(runDetection, 2000)
  }

  const stop = () => {
    setRunning(false)
    clearInterval(intervalRef.current)
  }

  useEffect(() => () => clearInterval(intervalRef.current), [])

  const handleUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = async (ev) => {
      setUploadedImage(ev.target.result)
      try {
        const { data } = await detectionAPI.detectVehiclesBase64(ev.target.result)
        setResult(data)
      } catch (err) { console.error(err) }
    }
    reader.readAsDataURL(file)
  }

  const density = result?.density_class || 'low'
  const densityColor = { low: 'text-green-400', medium: 'text-amber-400', high: 'text-crimson-400' }[density]
  const densityBg = { low: 'bg-green-500/10 border-green-500/25', medium: 'bg-amber-500/10 border-amber-500/25', high: 'bg-crimson-500/10 border-crimson-500/25' }[density]

  return (
    <div className="p-6 space-y-6 max-w-[1400px] mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="font-display font-bold text-2xl text-white">Live Traffic Monitor</h1>
          <p className="text-sm text-slate-500 mt-0.5">YOLOv8 real-time vehicle detection & tracking</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setMode('simulate')}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${mode === 'simulate' ? 'bg-crimson-600 text-white' : 'btn-ghost'}`}
          >
            Simulate Feed
          </button>
          <button
            onClick={() => setMode('upload')}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${mode === 'upload' ? 'bg-crimson-600 text-white' : 'btn-ghost'}`}
          >
            Upload Image
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-5">
        {/* Detection viewport */}
        <div className="lg:col-span-2 space-y-4">
          <div className="glass-card overflow-hidden">
            {/* Camera header bar */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/8">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${running ? 'bg-red-500 animate-pulse' : 'bg-slate-600'}`} />
                <span className="text-xs font-mono text-slate-400">
                  {running ? 'LIVE · CAM-01 · 1080p' : 'STANDBY'}
                </span>
              </div>
              <div className="flex items-center gap-2 text-xs font-mono text-slate-500">
                <Radio className="w-3 h-3" />
                {result?.inference_ms ? `${result.inference_ms}ms` : '--'}
              </div>
            </div>

            {/* Video frame area */}
            <div className="relative bg-slate-900 aspect-video flex items-center justify-center overflow-hidden">
              {mode === 'upload' && uploadedImage ? (
                <div className="relative w-full h-full">
                  <img src={uploadedImage} alt="uploaded" className="w-full h-full object-contain" />
                  {result?.detections?.map((det, i) => (
                    <DetectionBox key={i} det={det} frameW={640} frameH={480} />
                  ))}
                </div>
              ) : (
                <div className="relative w-full h-full bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
                  {/* Grid overlay */}
                  <div className="absolute inset-0 bg-grid-pattern opacity-20" />

                  {running ? (
                    <>
                      {/* Scanner line */}
                      <motion.div
                        animate={{ y: ['0%', '100%', '0%'] }}
                        transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
                        className="absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-crimson-500 to-transparent opacity-60 z-10"
                      />
                      {/* Mock vehicles grid */}
                      <div className="relative w-full h-full p-4">
                        <AnimatePresence>
                          {result?.detections?.slice(0, 8).map((det, i) => (
                            <motion.div
                              key={i}
                              initial={{ opacity: 0, scale: 0.8 }}
                              animate={{ opacity: 1, scale: 1 }}
                              exit={{ opacity: 0 }}
                              className="absolute border-2 rounded-lg flex items-end pb-1 pl-1"
                              style={{
                                left: `${10 + (i % 4) * 22}%`,
                                top: `${15 + Math.floor(i / 4) * 45}%`,
                                width: '18%', height: '35%',
                                borderColor: CLASS_COLORS[det.class] || '#fff',
                              }}
                            >
                              <span className="text-xs font-mono px-1 rounded"
                                style={{ backgroundColor: CLASS_COLORS[det.class], color: '#fff', fontSize: '9px' }}>
                                {det.class}
                              </span>
                            </motion.div>
                          ))}
                        </AnimatePresence>
                      </div>
                      <div className="absolute bottom-3 right-3 text-xs font-mono text-crimson-400 bg-black/40 px-2 py-1 rounded">
                        YOLO v8 · LIVE
                      </div>
                    </>
                  ) : (
                    <div className="text-center">
                      <div className="w-16 h-16 rounded-full border border-white/10 flex items-center justify-center mb-3 mx-auto">
                        <Play className="w-7 h-7 text-slate-600" />
                      </div>
                      <p className="text-sm text-slate-500">Click Start to begin detection</p>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Controls */}
            <div className="flex items-center gap-3 px-4 py-3 border-t border-white/8">
              {mode === 'simulate' ? (
                <>
                  {!running ? (
                    <button onClick={start} className="btn-primary py-2 px-5 flex items-center gap-2 text-sm">
                      <Play className="w-4 h-4" /> Start Detection
                    </button>
                  ) : (
                    <button onClick={stop}
                      className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-white py-2 px-5 rounded-xl text-sm font-medium transition-colors">
                      <Square className="w-4 h-4" /> Stop
                    </button>
                  )}
                </>
              ) : (
                <>
                  <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleUpload} />
                  <button onClick={() => fileRef.current?.click()} className="btn-primary py-2 px-5 flex items-center gap-2 text-sm">
                    <Upload className="w-4 h-4" /> Upload Image
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Live stats bar */}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`glass-card p-4 border ${densityBg}`}
            >
              <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
                <LiveStat label="Total Vehicles" value={result.total_vehicles} />
                <LiveStat label="Cars" value={result.vehicle_counts?.car || 0} color="text-blue-400" />
                <LiveStat label="Buses" value={result.vehicle_counts?.bus || 0} color="text-violet-400" />
                <LiveStat label="Trucks" value={result.vehicle_counts?.truck || 0} color="text-cyan-400" />
                <LiveStat label="Motorcycles" value={result.vehicle_counts?.motorcycle || 0} color="text-amber-400" />
                <div className="text-center">
                  <p className={`font-display font-bold text-xl ${densityColor}`}>{density.toUpperCase()}</p>
                  <p className="text-xs text-slate-500 mt-0.5">Density</p>
                </div>
              </div>
              {result.ambulance_detected && (
                <div className="mt-3 flex items-center gap-2 text-crimson-400 font-mono text-sm bg-crimson-600/10 px-3 py-2 rounded-lg border border-crimson-500/20 animate-pulse">
                  <Siren className="w-4 h-4" /> AMBULANCE DETECTED — EMERGENCY OVERRIDE ACTIVE
                </div>
              )}
            </motion.div>
          )}
        </div>

        {/* Detection log */}
        <div className="glass-card p-4 flex flex-col">
          <h3 className="font-display font-semibold text-sm text-white mb-3">Detection Log</h3>
          <div className="flex-1 overflow-y-auto space-y-2 max-h-[520px]">
            <AnimatePresence>
              {history.map((h, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="p-3 bg-white/3 rounded-xl border border-white/5 text-xs"
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-mono text-slate-500">{h.ts}</span>
                    <span className={`font-mono font-medium ${
                      h.density_class === 'high' ? 'text-crimson-400' :
                      h.density_class === 'medium' ? 'text-amber-400' : 'text-green-400'
                    }`}>{h.density_class?.toUpperCase()}</span>
                  </div>
                  <div className="flex items-center gap-3 flex-wrap">
                    <span className="text-white font-medium">{h.total_vehicles} vehicles</span>
                    {h.ambulance_detected && (
                      <span className="text-crimson-400 flex items-center gap-1">
                        <Siren className="w-3 h-3" /> AMB
                      </span>
                    )}
                    <span className="text-slate-500">{h.inference_ms}ms</span>
                    {h.mode === 'simulation' && <span className="text-slate-600">sim</span>}
                  </div>
                  <div className="flex gap-2 mt-1.5 flex-wrap">
                    {Object.entries(h.vehicle_counts || {}).map(([cls, cnt]) => (
                      <span key={cls} className="font-mono" style={{ color: CLASS_COLORS[cls] }}>
                        {cls}:{cnt}
                      </span>
                    ))}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            {history.length === 0 && (
              <p className="text-xs text-slate-600 text-center mt-8">Start detection to see logs</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
