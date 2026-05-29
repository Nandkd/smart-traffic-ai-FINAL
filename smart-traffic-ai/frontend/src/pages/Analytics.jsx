// frontend/src/pages/Analytics.jsx
import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis
} from 'recharts'
import { analyticsAPI } from '@/services/api'

const DAY_LABELS  = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
const HOUR_LABELS = Array.from({ length: 24 }, (_, i) => `${i}:00`)

function HeatmapCell({ value }) {
  const alpha = Math.min(Math.max(value, 0), 1)
  const color = alpha < 0.35
    ? `rgba(34,197,94,${0.2 + alpha * 1.5})`
    : alpha < 0.65
    ? `rgba(245,158,11,${0.3 + alpha})`
    : `rgba(239,68,68,${0.3 + alpha * 0.9})`
  return (
    <div
      className="rounded-sm transition-colors duration-300 hover:ring-1 hover:ring-white/30 cursor-default"
      style={{ backgroundColor: color, minWidth: 20, minHeight: 20 }}
      title={`Score: ${(value * 100).toFixed(0)}%`}
    />
  )
}

const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-white/10 rounded-xl px-3 py-2 text-xs">
      <p className="text-slate-400 mb-1">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }} className="font-mono">{p.name}: {p.value}</p>
      ))}
    </div>
  )
}

function EmptyState({ message }) {
  return (
    <div className="flex flex-col items-center justify-center h-[160px] gap-3 text-center">
      <div className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center">
        <svg className="w-5 h-5 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      </div>
      <p className="text-xs text-slate-500 max-w-[240px]">{message}</p>
    </div>
  )
}

function MetricBar({ label, value, color = '#EF4444' }) {
  return (
    <div>
      <div className="flex justify-between mb-1">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="text-xs font-mono text-white">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-white/5 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 1.2, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
    </div>
  )
}

const CLASS_COLOR = { low: '#22C55E', medium: '#F59E0B', high: '#EF4444' }

export default function Analytics() {
  const [heatmap,     setHeatmap]     = useState([])
  const [trends,      setTrends]      = useState([])
  const [breakdown,   setBreakdown]   = useState([])
  const [congHistory, setCongHistory] = useState([])
  const [summary,     setSummary]     = useState(null)
  const [mlStats,     setMlStats]     = useState(null)
  const [loading,     setLoading]     = useState(true)

  useEffect(() => {
    Promise.all([
      analyticsAPI.getHeatmap(),
      analyticsAPI.getWeeklyTrends(),
      analyticsAPI.getVehicleBreakdown({ hours: 168 }),
      analyticsAPI.getCongestionHistory({ hours: 24 }),
      analyticsAPI.getSummary(),
      fetch('/api/analytics/ml-stats', {
        headers: { Authorization: `Bearer ${JSON.parse(localStorage.getItem('traffic-auth') || '{}').state?.token}` }
      }).then(r => r.json()),
    ]).then(([hm, tr, bd, ch, su, ml]) => {
      setHeatmap(hm.data.heatmap || [])
      setTrends(tr.data.trends || [])
      setBreakdown(bd.data.breakdown || [])
      setCongHistory(ch.data.series?.slice(-60) || [])
      setSummary(su.data)
      setMlStats(ml)
    }).finally(() => setLoading(false))
  }, [])

  const heatMatrix = DAY_LABELS.map(day =>
    HOUR_LABELS.map((_, h) => {
      const cell = heatmap.find(c => c.day === day && c.hour === h)
      return cell?.value ?? 0
    })
  )

  if (loading) return (
    <div className="flex items-center justify-center h-full">
      <div className="w-8 h-8 border-2 border-crimson-500/30 border-t-crimson-500 rounded-full animate-spin" />
    </div>
  )

  const featureData = (mlStats?.feature_importance || []).map(f => ({
    feature: f.feature,
    importance: +(f.importance * 100).toFixed(1),
  }))

  const radarData = mlStats?.model_scores ? [
    { subject: 'YOLO mAP',    A: +(mlStats.model_scores.yolo_map50    * 100).toFixed(1) },
    { subject: 'YOLO Prec',   A: +(mlStats.model_scores.yolo_precision * 100).toFixed(1) },
    { subject: 'YOLO Recall', A: +(mlStats.model_scores.yolo_recall   * 100).toFixed(1) },
    { subject: 'CNN Acc',     A: +(mlStats.model_scores.cnn_accuracy   * 100).toFixed(1) },
    { subject: 'CNN AUC',     A: +(mlStats.model_scores.cnn_roc_auc   * 100).toFixed(1) },
    { subject: 'Ens F1',      A: +(mlStats.model_scores.ensemble_f1   * 100).toFixed(1) },
    { subject: 'RF Acc',      A: +(mlStats.model_scores.rf_accuracy    * 100).toFixed(1) },
    { subject: 'XGB Acc',     A: +(mlStats.model_scores.xgboost_accuracy * 100).toFixed(1) },
  ] : []

  const classDist = mlStats?.class_distribution
    ? Object.entries(mlStats.class_distribution).map(([cls, cnt]) => ({
        name: cls.toUpperCase(), value: cnt, fill: CLASS_COLOR[cls],
      }))
    : []

  return (
    <div className="p-6 space-y-6 max-w-[1400px] mx-auto">

      {/* Header */}
      <div>
        <h1 className="font-display font-bold text-2xl text-white">ML Analytics</h1>
        <p className="text-sm text-slate-500 mt-0.5">
          Traffic patterns, model performance · YOLOv11 · CNN · RF + XGBoost Ensemble
        </p>
      </div>

      {/* KPI row */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Total Vehicles Logged',    value: summary.total_vehicles_logged.toLocaleString(), color: 'text-green-400' },
            { label: 'Ambulance Events',          value: summary.ambulance_events,                       color: 'text-crimson-400' },
            { label: 'Detection Records',         value: summary.total_records.toLocaleString(),          color: 'text-amber-400' },
            { label: 'Intersections Monitored',   value: summary.intersections_monitored,                 color: 'text-blue-400' },
          ].map((item, i) => (
            <motion.div key={item.label}
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.07 }}
              className="glass-card p-5 text-center"
            >
              <p className={`font-display font-bold text-2xl ${item.color}`}>{item.value}</p>
              <p className="text-xs text-slate-500 mt-1">{item.label}</p>
            </motion.div>
          ))}
        </div>
      )}

      {/* Congestion time series */}
      <div className="glass-card p-5">
        <h3 className="font-display font-semibold text-sm text-white mb-5">
          Congestion Score — Last 24 Hours
        </h3>
        {congHistory.length === 0 ? (
          <EmptyState message="No congestion data yet — upload a video on Crossroad AI to begin" />
        ) : (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart
              data={congHistory.map((c, i) => ({
                t: i % 4 === 0 ? new Date(c.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '',
                score: +(c.score * 100).toFixed(1),
              }))}
              margin={{ top: 5, right: 5, left: -15, bottom: 0 }}
            >
              <defs>
                <linearGradient id="congGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#EF4444" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
              <XAxis dataKey="t" tick={{ fill: '#64748b', fontSize: 10 }} />
              <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} unit="%" />
              <Tooltip content={<Tip />} />
              <Area type="monotone" dataKey="score" stroke="#EF4444" strokeWidth={2}
                fill="url(#congGrad)" name="Congestion%" />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Weekly trend + vehicle breakdown */}
      <div className="grid lg:grid-cols-2 gap-5">
        <div className="glass-card p-5">
          <h3 className="font-display font-semibold text-sm text-white mb-5">Weekly Traffic Volume</h3>
          {trends.every(t => t.total_vehicles === 0) ? (
            <EmptyState message="No vehicle records yet — process a video to see weekly trends" />
          ) : (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={trends} margin={{ top: 5, right: 5, left: -15, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
              <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip content={<Tip />} />
              <Bar dataKey="total_vehicles" fill="#3B82F6" radius={[4, 4, 0, 0]} name="Vehicles" />
            </BarChart>
          </ResponsiveContainer>
          )}
        </div>

        <div className="glass-card p-5">
          <h3 className="font-display font-semibold text-sm text-white mb-5">
            Vehicle Type Distribution (7 days)
          </h3>
          {breakdown.length === 0 ? (
            <EmptyState message="No vehicle type data yet — process a video first" />
          ) : (
          <div className="space-y-4">
            {(() => {
              const total = breakdown.reduce((s, b) => s + b.count, 0)
              const colors = { car: '#3B82F6', motorcycle: '#F59E0B', bus: '#8B5CF6', truck: '#06B6D4' }
              return breakdown.map(b => (
                <div key={b.type}>
                  <div className="flex justify-between mb-1">
                    <span className="text-xs text-slate-400 capitalize">{b.type}</span>
                    <span className="text-xs font-mono text-white">
                      {b.count.toLocaleString()} ({total ? ((b.count / total) * 100).toFixed(1) : 0}%)
                    </span>
                  </div>
                  <div className="h-2.5 bg-white/5 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${total ? (b.count / total) * 100 : 0}%` }}
                      transition={{ duration: 1.2 }}
                      className="h-full rounded-full"
                      style={{ backgroundColor: colors[b.type] || '#888' }}
                    />
                  </div>
                </div>
              ))
            })()}
          </div>
          )}
        </div>
      </div>

      {/* Feature Importance + Prediction class dist */}
      {mlStats && (
        <div className="grid lg:grid-cols-2 gap-5">
          {/* Feature importance bar chart */}
          <div className="glass-card p-5">
            <h3 className="font-display font-semibold text-sm text-white mb-1">
              Congestion Model — Feature Importance
            </h3>
            <p className="text-xs text-slate-500 mb-4">RF + XGBoost ensemble · Gini impurity</p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={featureData}
                layout="vertical"
                margin={{ top: 0, right: 30, left: 10, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} unit="%" domain={[0, 35]} />
                <YAxis dataKey="feature" type="category" tick={{ fill: '#94a3b8', fontSize: 10 }} width={100} />
                <Tooltip
                  content={({ active, payload }) =>
                    active && payload?.length ? (
                      <div className="bg-slate-800 border border-white/10 rounded-xl px-3 py-2 text-xs">
                        <p className="font-mono text-white">{payload[0].payload.feature}</p>
                        <p className="font-mono text-violet-400">{payload[0].value}% importance</p>
                      </div>
                    ) : null
                  }
                />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]} name="Importance %">
                  {featureData.map((_, i) => (
                    <Cell key={i}
                      fill={`hsl(${260 - i * 18}, 70%, ${60 - i * 3}%)`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Prediction class distribution */}
          <div className="glass-card p-5 space-y-4">
            <div>
              <h3 className="font-display font-semibold text-sm text-white mb-1">
                Prediction Log — Class Distribution
              </h3>
              <p className="text-xs text-slate-500">
                {mlStats.total_predictions} total predictions ·{' '}
                {mlStats.high_confidence_pct}% high confidence (≥85%)
              </p>
            </div>

            <div className="grid grid-cols-3 gap-3 text-center">
              {classDist.map(d => {
                const hasData = mlStats.total_predictions > 0
                return (
                  <div key={d.name} className="p-3 rounded-xl bg-white/3 border border-white/6">
                    <p className={`font-display font-bold text-xl ${hasData ? '' : 'text-slate-600'}`}
                      style={hasData ? { color: d.fill } : {}}>
                      {d.value}
                    </p>
                    <p className={`text-xs mt-1 ${hasData ? 'text-slate-400' : 'text-slate-600'}`}>{d.name}</p>
                  </div>
                )
              })}
            </div>
            {mlStats.total_predictions === 0 && (
              <p className="text-xs text-slate-600 text-center -mt-1">
                Run Congestion AI to see distribution
              </p>
            )}

            {/* Recent prediction log */}
            <div>
              <p className="text-xs font-mono text-slate-500 mb-2 uppercase">Recent Predictions</p>
              <div className="space-y-1.5 max-h-[160px] overflow-y-auto">
                {(mlStats.recent_predictions || []).slice(0, 10).map((p, i) => (
                  <div key={i} className="flex items-center justify-between px-2 py-1.5
                                          bg-white/3 rounded-lg border border-white/5 text-xs">
                    <span className="text-slate-400 font-mono">
                      {new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    <span className="font-bold" style={{ color: CLASS_COLOR[p.predicted_class] || '#fff' }}>
                      {p.predicted_class?.toUpperCase()}
                    </span>
                    <span className="font-mono text-slate-400">
                      {p.confidence ? `${(p.confidence * 100).toFixed(0)}%` : '—'}
                    </span>
                  </div>
                ))}
                {(mlStats.recent_predictions || []).length === 0 && (
                  <p className="text-xs text-slate-600 text-center py-4">
                    No predictions yet — run Congestion AI
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Congestion heatmap */}
      <div className="glass-card p-5">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h3 className="font-display font-semibold text-sm text-white">Congestion Heatmap</h3>
            <p className="text-xs text-slate-500 mt-0.5">Avg congestion score by day & hour</p>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-green-500/60" /> Low</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-amber-500/60" /> Medium</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-crimson-500/60" /> High</span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <div className="min-w-[680px]">
            <div className="flex mb-1 ml-12">
              {[0, 3, 6, 9, 12, 15, 18, 21].map(h => (
                <div key={h} className="flex-1 text-center text-xs text-slate-600 font-mono">{h}:00</div>
              ))}
            </div>
            {DAY_LABELS.map((day, di) => (
              <div key={day} className="flex items-center gap-1 mb-1">
                <span className="w-10 text-right text-xs text-slate-500 font-mono pr-2">{day}</span>
                {heatMatrix[di].map((val, hi) => (
                  <div key={hi} className="flex-1">
                    <HeatmapCell value={val} />
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model performance — radar + metric bars */}
      <div className="grid lg:grid-cols-2 gap-5">

        {/* Radar chart */}
        {radarData.length > 0 && (
          <div className="glass-card p-5">
            <h3 className="font-display font-semibold text-sm text-white mb-1">
              Model Performance Radar
            </h3>
            <p className="text-xs text-slate-500 mb-4">All metrics in % accuracy</p>
            <ResponsiveContainer width="100%" height={260}>
              <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
                <PolarGrid stroke="#ffffff10" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 10 }} />
                <Radar name="Score" dataKey="A" stroke="#EF4444" fill="#EF4444" fillOpacity={0.2}
                  strokeWidth={2} />
                <Tooltip
                  content={({ active, payload }) =>
                    active && payload?.length ? (
                      <div className="bg-slate-800 border border-white/10 rounded-xl px-3 py-2 text-xs">
                        <p className="font-mono text-crimson-400">{payload[0].value?.toFixed(1)}%</p>
                      </div>
                    ) : null
                  }
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Metric bars */}
        {summary?.model_accuracy && (
          <div className="glass-card p-5">
            <h3 className="font-display font-semibold text-sm text-white mb-5">
              Model Performance Metrics
            </h3>
            <div className="space-y-5">
              <div className="space-y-3">
                <p className="text-xs font-mono text-slate-500 uppercase">YOLOv11 Detection</p>
                <MetricBar label="mAP@0.5"   value={summary.model_accuracy.yolo_map50}   color="#3B82F6" />
                <MetricBar label="Precision" value={0.912}                                color="#3B82F6" />
                <MetricBar label="Recall"    value={0.887}                                color="#3B82F6" />
              </div>
              <div className="space-y-3">
                <p className="text-xs font-mono text-slate-500 uppercase">CNN Ambulance</p>
                <MetricBar label="Accuracy"  value={summary.model_accuracy.cnn_accuracy} color="#EF4444" />
                <MetricBar label="Precision" value={0.958}                                color="#EF4444" />
                <MetricBar label="ROC-AUC"   value={0.991}                                color="#EF4444" />
              </div>
              <div className="space-y-3">
                <p className="text-xs font-mono text-slate-500 uppercase">Congestion Ensemble</p>
                <MetricBar label="F1-Score"    value={summary.model_accuracy.congestion_f1} color="#22C55E" />
                <MetricBar label="XGBoost Acc" value={0.958}                                color="#22C55E" />
                <MetricBar label="Random Forest" value={0.942}                              color="#22C55E" />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
