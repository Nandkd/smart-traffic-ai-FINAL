// frontend/src/components/charts/index.jsx
/**
 * Reusable pre-styled chart components built on Recharts.
 * All charts use the dark premium theme.
 */

import React from 'react'
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

// ── Shared theme ───────────────────────────────────────────────
const GRID_COLOR   = 'rgba(255,255,255,0.06)'
const TICK_COLOR   = '#64748b'
const COLORS       = ['#EF4444', '#3B82F6', '#22C55E', '#F59E0B', '#8B5CF6', '#06B6D4']

// ── Custom Tooltip ─────────────────────────────────────────────
export function DarkTooltip({ active, payload, label, unit = '' }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-white/10 rounded-xl px-3 py-2.5 text-xs shadow-xl">
      {label && <p className="text-slate-400 mb-2 font-mono">{label}</p>}
      {payload.map((p, i) => (
        <p key={i} className="font-mono" style={{ color: p.color }}>
          {p.name}: <span className="text-white">{typeof p.value === 'number' ? p.value.toLocaleString() : p.value}{unit}</span>
        </p>
      ))}
    </div>
  )
}

// ── Area Chart ─────────────────────────────────────────────────
export function TrafficAreaChart({ data, dataKey = 'value', color = '#EF4444', xKey = 'label', height = 200, unit = '' }) {
  const gradId = `grad-${dataKey}`
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={color} stopOpacity={0.28} />
            <stop offset="95%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
        <XAxis dataKey={xKey} tick={{ fill: TICK_COLOR, fontSize: 11 }} />
        <YAxis tick={{ fill: TICK_COLOR, fontSize: 11 }} />
        <Tooltip content={<DarkTooltip unit={unit} />} />
        <Area type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2}
          fill={`url(#${gradId})`} dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  )
}

// ── Multi-line Chart ───────────────────────────────────────────
export function MultiLineChart({ data, lines, xKey = 'label', height = 200 }) {
  // lines: [{ dataKey, color, name }]
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
        <XAxis dataKey={xKey} tick={{ fill: TICK_COLOR, fontSize: 11 }} />
        <YAxis tick={{ fill: TICK_COLOR, fontSize: 11 }} />
        <Tooltip content={<DarkTooltip />} />
        <Legend wrapperStyle={{ color: TICK_COLOR, fontSize: 11 }} />
        {lines.map(l => (
          <Line key={l.dataKey} type="monotone" dataKey={l.dataKey}
            stroke={l.color || COLORS[0]} strokeWidth={2}
            dot={false} name={l.name || l.dataKey} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

// ── Bar Chart ──────────────────────────────────────────────────
export function TrafficBarChart({ data, dataKey = 'value', xKey = 'label', color = '#3B82F6', height = 200, unit = '' }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
        <XAxis dataKey={xKey} tick={{ fill: TICK_COLOR, fontSize: 11 }} />
        <YAxis tick={{ fill: TICK_COLOR, fontSize: 11 }} />
        <Tooltip content={<DarkTooltip unit={unit} />} />
        <Bar dataKey={dataKey} fill={color} radius={[4, 4, 0, 0]} name={dataKey} />
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Donut / Pie Chart ──────────────────────────────────────────
export function DonutChart({ data, nameKey = 'name', valueKey = 'value', height = 220 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie data={data} cx="50%" cy="50%"
          innerRadius="55%" outerRadius="80%"
          dataKey={valueKey} nameKey={nameKey}
          paddingAngle={3} strokeWidth={0}>
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip content={<DarkTooltip />} />
      </PieChart>
    </ResponsiveContainer>
  )
}

// ── Radar Chart (model performance) ───────────────────────────
export function ModelRadarChart({ data, height = 260 }) {
  // data: [{ metric, rf, xgb, lr }]
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RadarChart data={data}>
        <PolarGrid stroke={GRID_COLOR} />
        <PolarAngleAxis dataKey="metric" tick={{ fill: TICK_COLOR, fontSize: 10 }} />
        <PolarRadiusAxis tick={{ fill: TICK_COLOR, fontSize: 9 }} domain={[0, 1]} />
        <Radar name="Random Forest" dataKey="rf"  stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.25} />
        <Radar name="XGBoost"       dataKey="xgb" stroke="#EF4444" fill="#EF4444" fillOpacity={0.25} />
        <Radar name="Log. Reg."     dataKey="lr"  stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.2}  />
        <Legend wrapperStyle={{ color: TICK_COLOR, fontSize: 11 }} />
        <Tooltip content={<DarkTooltip />} />
      </RadarChart>
    </ResponsiveContainer>
  )
}

// ── Congestion score gauge (SVG) ───────────────────────────────
export function CongestionGauge({ score = 0.5, size = 120 }) {
  const r = size * 0.38
  const cx = size / 2
  const cy = size / 2
  const circ = 2 * Math.PI * r
  const half = circ / 2              // we use only top half
  const offset = half * (1 - score)
  const color = score > 0.65 ? '#EF4444' : score > 0.35 ? '#F59E0B' : '#22C55E'

  return (
    <svg width={size} height={size * 0.6} viewBox={`0 0 ${size} ${size * 0.6}`}>
      {/* Track */}
      <path
        d={`M ${cx - r},${cy} A ${r},${r} 0 0,1 ${cx + r},${cy}`}
        fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={size * 0.07}
        strokeLinecap="round"
      />
      {/* Value arc */}
      <path
        d={`M ${cx - r},${cy} A ${r},${r} 0 0,1 ${cx + r},${cy}`}
        fill="none" stroke={color} strokeWidth={size * 0.07}
        strokeDasharray={`${half} ${half}`}
        strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 1s ease' }}
      />
      <text x={cx} y={cy - 4} textAnchor="middle" fill="white"
        fontSize={size * 0.18} fontWeight="bold" fontFamily="Syne, sans-serif">
        {(score * 100).toFixed(0)}%
      </text>
      <text x={cx} y={cy + size * 0.1} textAnchor="middle" fill="#64748b"
        fontSize={size * 0.1} fontFamily="DM Sans, sans-serif">
        congestion
      </text>
    </svg>
  )
}
