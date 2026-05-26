// frontend/src/pages/AmbulanceDetect.jsx
import React, { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Siren, Upload, CheckCircle, XCircle, AlertTriangle, Camera } from 'lucide-react'
import { detectionAPI, predictionAPI, signalsAPI } from '@/services/api'

function PulsingRing({ color }) {
  return (
    <div className="relative flex items-center justify-center w-32 h-32 mx-auto">
      {[1, 2, 3].map(i => (
        <motion.div
          key={i}
          className="absolute inset-0 rounded-full border-2"
          style={{ borderColor: color }}
          animate={{ scale: [1, 1.4 + i * 0.3], opacity: [0.8, 0] }}
          transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.4, ease: 'easeOut' }}
        />
      ))}
      <div className="w-20 h-20 rounded-full flex items-center justify-center border-2"
        style={{ borderColor: color, backgroundColor: `${color}15` }}>
        <Siren className="w-9 h-9" style={{ color }} />
      </div>
    </div>
  )
}

export default function AmbulanceDetect() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [override, setOverride] = useState(null)
  const [history, setHistory] = useState([])
  const [previewSrc, setPreviewSrc] = useState(null)
  const fileRef = useRef(null)

  const runDetection = async (file) => {
    setLoading(true)
    setResult(null)
    try {
      const { data } = await detectionAPI.detectAmbulance(file ? new FormData() : undefined)
      setResult(data)
      setHistory(prev => [{ ...data, ts: new Date().toLocaleTimeString() }, ...prev.slice(0, 19)])

      if (data.ambulance_detected) {
        // Trigger emergency override on signal 1, north lane
        const overrideRes = await signalsAPI.emergency(1, 'north')
        setOverride(overrideRes.data.signal)
      }
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleFile = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    setPreviewSrc(url)
    runDetection(file)
  }

  const handleSimulate = () => {
    setPreviewSrc(null)
    runDetection(null)
  }

  const detected = result?.ambulance_detected
  const confidence = result?.confidence || 0

  return (
    <div className="p-6 space-y-6 max-w-[1200px] mx-auto">
      {/* Header */}
      <div>
        <h1 className="font-display font-bold text-2xl text-white">Ambulance Detection AI</h1>
        <p className="text-sm text-slate-500 mt-0.5">
          CNN classifier · 96.2% accuracy · &lt;12ms inference · Emergency signal override
        </p>
      </div>

      <div className="grid lg:grid-cols-3 gap-5">
        {/* Detection panel */}
        <div className="lg:col-span-2 space-y-4">
          {/* Result visualizer */}
          <div className="glass-card p-8">
            <AnimatePresence mode="wait">
              {loading && (
                <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex flex-col items-center gap-4 py-8">
                  <div className="w-12 h-12 border-2 border-crimson-500/30 border-t-crimson-500 rounded-full animate-spin" />
                  <p className="text-sm text-slate-400 font-mono">Running CNN inference...</p>
                </motion.div>
              )}

              {!loading && !result && (
                <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="flex flex-col items-center gap-4 py-8 text-center">
                  <div className="w-20 h-20 rounded-full border border-white/10 flex items-center justify-center">
                    <Siren className="w-9 h-9 text-slate-600" />
                  </div>
                  <div>
                    <p className="text-base font-medium text-white">CNN Ambulance Classifier</p>
                    <p className="text-sm text-slate-500 mt-1">Upload an image or run a simulation</p>
                  </div>
                </motion.div>
              )}

              {!loading && result && (
                <motion.div key="result" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                  className="flex flex-col items-center gap-6">
                  {previewSrc && (
                    <img src={previewSrc} alt="input"
                      className="max-h-40 rounded-xl object-contain border border-white/10" />
                  )}

                  <PulsingRing color={detected ? '#EF4444' : '#22C55E'} />

                  <div className="text-center">
                    <p className="font-display font-bold text-3xl mb-2"
                      style={{ color: detected ? '#EF4444' : '#22C55E' }}>
                      {detected ? 'AMBULANCE DETECTED' : 'NO AMBULANCE'}
                    </p>
                    <p className="text-slate-400 text-sm mb-1">
                      Confidence: <span className="text-white font-mono">{(confidence * 100).toFixed(2)}%</span>
                    </p>
                    <p className="text-slate-500 text-xs font-mono">
                      {result.inference_ms}ms · {result.model}
                    </p>
                  </div>

                  {/* Confidence bar */}
                  <div className="w-full max-w-sm">
                    <div className="flex justify-between text-xs text-slate-500 mb-1.5">
                      <span>Non-Ambulance</span>
                      <span>Ambulance</span>
                    </div>
                    <div className="h-3 bg-white/5 rounded-full overflow-hidden relative">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${confidence * 100}%` }}
                        transition={{ duration: 0.9, ease: 'easeOut' }}
                        className="h-full rounded-full"
                        style={{ backgroundColor: detected ? '#EF4444' : '#22C55E' }}
                      />
                    </div>
                  </div>

                  {/* Action badge */}
                  <div className={`px-4 py-2 rounded-xl border text-sm font-mono ${
                    detected
                      ? 'bg-crimson-600/15 border-crimson-600/30 text-crimson-400 animate-pulse'
                      : 'bg-green-500/10 border-green-500/20 text-green-400'
                  }`}>
                    {result.action}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Controls */}
          <div className="flex gap-3">
            <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleFile} />
            <button onClick={() => fileRef.current?.click()} className="btn-ghost flex items-center gap-2 flex-1">
              <Upload className="w-4 h-4" /> Upload Image
            </button>
            <button onClick={handleSimulate} disabled={loading} className="btn-primary flex items-center gap-2 flex-1 justify-center disabled:opacity-60">
              <Camera className="w-4 h-4" /> Simulate Detection
            </button>
          </div>

          {/* Emergency override status */}
          <AnimatePresence>
            {override && detected && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="glass-card p-4 border border-crimson-600/30 bg-crimson-600/8"
              >
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-crimson-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-crimson-300">Emergency Override Active</p>
                    <p className="text-xs text-slate-400 mt-1">
                      Signal <strong className="text-white">{override.location_name}</strong> — North lane:
                      <span className="font-mono text-green-400 ml-1">90s GREEN</span>
                      , all other lanes:
                      <span className="font-mono text-crimson-400 ml-1">5s</span>
                    </p>
                    <button
                      onClick={async () => { await signalsAPI.reset(1); setOverride(null) }}
                      className="mt-2 text-xs text-slate-500 hover:text-white underline transition-colors"
                    >
                      Reset to normal operation
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Detection history */}
        <div className="glass-card p-4 flex flex-col">
          <h3 className="font-display font-semibold text-sm text-white mb-3">Detection Log</h3>
          <div className="flex-1 overflow-y-auto space-y-2 max-h-[480px]">
            {history.map((h, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: 8 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-center gap-3 p-3 bg-white/3 rounded-xl border border-white/5"
              >
                {h.ambulance_detected
                  ? <CheckCircle className="w-4 h-4 text-crimson-400 flex-shrink-0" />
                  : <XCircle className="w-4 h-4 text-green-400 flex-shrink-0" />
                }
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-white truncate">
                    {h.ambulance_detected ? 'Ambulance' : 'No Ambulance'}
                  </p>
                  <p className="text-xs text-slate-500 font-mono mt-0.5">
                    {(h.confidence * 100).toFixed(1)}% · {h.ts}
                  </p>
                </div>
              </motion.div>
            ))}
            {history.length === 0 && (
              <p className="text-xs text-slate-600 text-center mt-8">No detections yet</p>
            )}
          </div>

          {/* Stats summary */}
          {history.length > 0 && (
            <div className="border-t border-white/8 pt-3 mt-3 grid grid-cols-2 gap-3 text-center">
              <div>
                <p className="text-lg font-display font-bold text-crimson-400">
                  {history.filter(h => h.ambulance_detected).length}
                </p>
                <p className="text-xs text-slate-500">Ambulances</p>
              </div>
              <div>
                <p className="text-lg font-display font-bold text-white">{history.length}</p>
                <p className="text-xs text-slate-500">Total Checks</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Model architecture info */}
      <div className="glass-card p-5">
        <h3 className="font-display font-semibold text-sm text-white mb-4">CNN Architecture</h3>
        <div className="flex items-center gap-2 flex-wrap font-mono text-xs overflow-x-auto pb-2">
          {[
            { label: 'Input\n3×224×224', color: 'bg-slate-700' },
            { label: 'Conv Block 1\n32 filters', color: 'bg-blue-900/60' },
            { label: 'Conv Block 2\n64 filters', color: 'bg-blue-800/60' },
            { label: 'Conv Block 3\n128 filters', color: 'bg-violet-900/60' },
            { label: 'Conv Block 4\n256 filters', color: 'bg-violet-800/60' },
            { label: 'Avg Pool\n4×4', color: 'bg-slate-700' },
            { label: 'FC 512\nDropout 0.4', color: 'bg-crimson-900/60' },
            { label: 'FC 128\nDropout 0.2', color: 'bg-crimson-800/60' },
            { label: 'Output\n2 classes', color: 'bg-green-900/60' },
          ].map((layer, i) => (
            <React.Fragment key={i}>
              <div className={`${layer.color} border border-white/10 rounded-lg px-3 py-2 text-center whitespace-pre-line min-w-[90px]`}>
                {layer.label}
              </div>
              {i < 8 && <span className="text-slate-600">→</span>}
            </React.Fragment>
          ))}
        </div>
        <div className="flex gap-6 mt-4 text-xs">
          {[
            ['Parameters', '~4.2M'],
            ['Input Size', '224×224×3'],
            ['Accuracy', '96.2%'],
            ['ROC-AUC', '0.991'],
            ['Inference', '<12ms'],
            ['Optimizer', 'AdamW'],
          ].map(([k, v]) => (
            <div key={k}>
              <p className="text-slate-500">{k}</p>
              <p className="text-white font-medium mt-0.5">{v}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
