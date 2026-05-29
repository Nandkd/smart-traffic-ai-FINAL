// frontend/src/pages/CongestionPredict.jsx
import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, TrendingUp, Clock, CloudRain, AlertTriangle, ChevronRight } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { predictionAPI } from '@/services/api'

const DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

const defaultForm = {
  vehicle_count: 0,
  car_count: 0,
  bus_count: 0,
  truck_count: 0,
  motorcycle_count: 0,
  hour: new Date().getHours(),
  day_of_week: new Date().getDay(),
  rain_intensity: 0,
  visibility: 1.0,
  incident_nearby: 0,
}

function Slider({ label, min, max, step = 1, value, onChange, unit = '' }) {
  return (
    <div>
      <div className="flex justify-between mb-1.5">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="text-xs font-mono text-white">{value}{unit}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-white/10 rounded-full appearance-none cursor-pointer
                   [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4
                   [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:bg-crimson-500 [&::-webkit-slider-thumb]:cursor-pointer"
      />
    </div>
  )
}

function ProbBar({ label, value, color }) {
  return (
    <div>
      <div className="flex justify-between mb-1">
        <span className="text-sm font-medium text-white capitalize">{label}</span>
        <span className="text-sm font-mono" style={{ color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-3 bg-white/5 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
    </div>
  )
}

export default function CongestionPredict() {
  const [form, setForm] = useState(defaultForm)
  const [result, setResult] = useState(null)
  const [peakHours, setPeakHours] = useState([])
  const [loading, setLoading] = useState(false)
  const [peakLoading, setPeakLoading] = useState(true)

  const setField = (key) => (val) => setForm(prev => ({ ...prev, [key]: val }))

  useEffect(() => {
    predictionAPI.getPeakHours({ day: form.day_of_week }).then(res => {
      setPeakHours(res.data.hourly_forecast || [])
    }).finally(() => setPeakLoading(false))
  }, [form.day_of_week])

  const predict = async () => {
    setLoading(true)
    try {
      const { data } = await predictionAPI.predictCongestion({
        ...form,
        intersection_id: 1,
      })
      setResult(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const densityColor = {
    low: '#22C55E', medium: '#F59E0B', high: '#EF4444'
  }

  const peakChartData = peakHours.map(h => ({
    hour: `${h.hour}:00`,
    score: +(h.congestion_score * 100).toFixed(1),
    class: h.predicted_class,
  }))

  return (
    <div className="p-6 space-y-6 max-w-[1300px] mx-auto">
      {/* Header */}
      <div>
        <h1 className="font-display font-bold text-2xl text-white">Congestion AI Prediction</h1>
        <p className="text-sm text-slate-500 mt-0.5">
          Random Forest + XGBoost + Logistic Regression ensemble · 96.4% accuracy
        </p>
      </div>

      <div className="grid lg:grid-cols-5 gap-5">
        {/* Input form */}
        <div className="lg:col-span-2 space-y-4">
          <div className="glass-card p-5">
            <h3 className="font-display font-semibold text-sm text-white mb-5 flex items-center gap-2">
              <Brain className="w-4 h-4 text-violet-400" /> Input Features
            </h3>

            <div className="space-y-5">
              <div>
                <p className="text-xs font-mono text-slate-500 uppercase tracking-wider mb-3">Vehicle Counts</p>
                <div className="space-y-4">
                  <Slider label="Total Vehicles" min={0} max={200} value={form.vehicle_count} onChange={setField('vehicle_count')} />
                  <Slider label="Cars" min={0} max={120} value={form.car_count} onChange={setField('car_count')} />
                  <Slider label="Buses" min={0} max={40} value={form.bus_count} onChange={setField('bus_count')} />
                  <Slider label="Trucks" min={0} max={40} value={form.truck_count} onChange={setField('truck_count')} />
                  <Slider label="Motorcycles" min={0} max={80} value={form.motorcycle_count} onChange={setField('motorcycle_count')} />
                </div>
              </div>

              <div className="border-t border-white/8 pt-4">
                <p className="text-xs font-mono text-slate-500 uppercase tracking-wider mb-3">Time Context</p>
                <div className="space-y-4">
                  <Slider label="Hour of Day" min={0} max={23} value={form.hour} onChange={setField('hour')} unit=":00" />
                  <div>
                    <label className="text-xs text-slate-400 block mb-2">Day of Week</label>
                    <select
                      value={form.day_of_week}
                      onChange={e => setField('day_of_week')(parseInt(e.target.value))}
                      className="w-full bg-slate-800 border border-white/10 rounded-xl px-3 py-2.5 text-sm text-white
                                 focus:outline-none focus:border-crimson-500/50"
                    >
                      {DAYS.map((d, i) => <option key={d} value={i}>{d}</option>)}
                    </select>
                  </div>
                </div>
              </div>

              <div className="border-t border-white/8 pt-4">
                <p className="text-xs font-mono text-slate-500 uppercase tracking-wider mb-3">Conditions</p>
                <div className="space-y-4">
                  <Slider label="Rain Intensity" min={0} max={1} step={0.05} value={form.rain_intensity} onChange={setField('rain_intensity')} />
                  <Slider label="Visibility" min={0.1} max={1} step={0.05} value={form.visibility} onChange={setField('visibility')} />
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-400">Incident Nearby</span>
                    <button
                      onClick={() => setField('incident_nearby')(form.incident_nearby ? 0 : 1)}
                      className={`w-10 h-5 rounded-full transition-colors ${form.incident_nearby ? 'bg-crimson-600' : 'bg-white/10'} relative`}
                    >
                      <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform ${form.incident_nearby ? 'translate-x-5' : 'translate-x-0.5'}`} />
                    </button>
                  </div>
                </div>
              </div>

              <button
                onClick={predict}
                disabled={loading}
                className="w-full btn-primary flex items-center justify-center gap-2 disabled:opacity-60"
              >
                {loading
                  ? <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  : <><Brain className="w-4 h-4" /> Run ML Prediction</>
                }
              </button>
            </div>
          </div>
        </div>

        {/* Results */}
        <div className="lg:col-span-3 space-y-4">
          <AnimatePresence mode="wait">
            {result && (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-card p-6"
              >
                {/* Predicted class */}
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <p className="text-xs font-mono text-slate-500 uppercase tracking-wider mb-1">Ensemble Prediction</p>
                    <p className="font-display font-bold text-4xl"
                      style={{ color: densityColor[result.predicted_class] }}>
                      {result.predicted_class?.toUpperCase()}
                    </p>
                    <p className="text-sm text-slate-400 mt-1">
                      Confidence: <span className="text-white font-mono">{(result.ensemble_confidence * 100).toFixed(1)}%</span>
                    </p>
                  </div>
                  <div className={`w-20 h-20 rounded-2xl flex items-center justify-center border-2`}
                    style={{ borderColor: densityColor[result.predicted_class], backgroundColor: `${densityColor[result.predicted_class]}15` }}>
                    <span className="text-3xl">
                      {result.predicted_class === 'high' ? '🔴' : result.predicted_class === 'medium' ? '🟡' : '🟢'}
                    </span>
                  </div>
                </div>

                {/* Probability bars */}
                <div className="space-y-3 mb-6">
                  <p className="text-xs font-mono text-slate-500 uppercase tracking-wider">Class Probabilities</p>
                  <ProbBar label="Low Traffic" value={result.probabilities?.low || 0} color="#22C55E" />
                  <ProbBar label="Medium Traffic" value={result.probabilities?.medium || 0} color="#F59E0B" />
                  <ProbBar label="High Traffic" value={result.probabilities?.high || 0} color="#EF4444" />
                </div>

                {/* Per-model breakdown */}
                {result.individual_models && (
                  <div className="border-t border-white/8 pt-4">
                    <p className="text-xs font-mono text-slate-500 uppercase tracking-wider mb-3">Individual Models</p>
                    <div className="grid grid-cols-3 gap-3">
                      {Object.entries(result.individual_models).map(([model, pred]) => (
                        <div key={model} className="bg-white/3 rounded-xl p-3 text-center border border-white/6">
                          <p className="text-xs text-slate-400 mb-1 capitalize">{model.replace('_', ' ')}</p>
                          <p className="font-mono font-semibold text-sm" style={{ color: densityColor[pred.class] }}>
                            {pred.class?.toUpperCase()}
                          </p>
                          <p className="text-xs text-slate-500 mt-0.5 font-mono">{(pred.confidence * 100).toFixed(1)}%</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recommendation */}
                <div className="mt-4 p-3 rounded-xl bg-violet-500/8 border border-violet-500/15">
                  <p className="text-xs text-violet-300 font-mono">
                    {result.predicted_class === 'high'
                      ? '⚠ Recommendation: Extend green phase by 40s on busiest lane. Consider diverting secondary routes.'
                      : result.predicted_class === 'medium'
                      ? '⚡ Recommendation: Apply 20s green extension. Monitor for escalation.'
                      : '✓ Recommendation: Normal cycle timing. No intervention required.'}
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {!result && (
            <div className="glass-card p-8 flex flex-col items-center justify-center text-center min-h-[200px]">
              <Brain className="w-10 h-10 text-slate-700 mb-3" />
              <p className="text-sm text-slate-500">Adjust input features and click Run ML Prediction</p>
            </div>
          )}

          {/* Peak hour forecast chart */}
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-display font-semibold text-sm text-white">Peak Hour Forecast</h3>
                <p className="text-xs text-slate-500 mt-0.5">{DAYS[form.day_of_week]} — all 24 hours</p>
              </div>
              <Clock className="w-4 h-4 text-slate-600" />
            </div>
            {peakLoading ? (
              <div className="h-[180px] flex items-center justify-center">
                <div className="w-6 h-6 border-2 border-crimson-500/30 border-t-crimson-500 rounded-full animate-spin" />
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={peakChartData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
                  <XAxis dataKey="hour" tick={{ fill: '#64748b', fontSize: 9 }} interval={2} />
                  <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} unit="%" />
                  <Tooltip
                    content={({ active, payload, label }) => active && payload?.length ? (
                      <div className="bg-slate-800 border border-white/10 rounded-xl px-3 py-2 text-xs">
                        <p className="text-slate-400">{label}</p>
                        <p className="font-mono text-white">{payload[0]?.value?.toFixed(1)}%</p>
                        <p style={{ color: densityColor[payload[0]?.payload?.class] || '#fff' }}>
                          {payload[0]?.payload?.class}
                        </p>
                      </div>
                    ) : null}
                  />
                  <Bar dataKey="score" radius={[3, 3, 0, 0]} name="Congestion%">
                    {peakChartData.map((entry, i) => (
                      <Cell key={i} fill={densityColor[entry.class] || '#3B82F6'} fillOpacity={0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
