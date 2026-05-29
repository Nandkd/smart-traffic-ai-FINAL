// frontend/src/pages/Dashboard.jsx
import React, { useEffect, useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import {
  Car, Bus, TruckIcon, Bike, Siren, Activity,
  TrendingUp, TrendingDown, Signal, Clock, AlertTriangle
} from 'lucide-react'
import { trafficAPI, analyticsAPI, crossroadAPI } from '@/services/api'

// ── Reusable stat card ──────────────────────────────────────────
function StatCard({ icon: Icon, label, value, subtext, trend, color = 'blue', delay = 0 }) {
  const colors = {
    red: 'text-crimson-400 bg-crimson-500/10 border-crimson-500/20',
    green: 'text-green-400 bg-green-500/10 border-green-500/20',
    blue: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
    amber: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    violet: 'text-violet-400 bg-violet-500/10 border-violet-500/20',
  }
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="stat-card"
    >
      <div className="flex items-start justify-between">
        <div className={`w-10 h-10 rounded-xl border flex items-center justify-center ${colors[color]}`}>
          <Icon className="w-4.5 h-4.5" />
        </div>
        {trend !== undefined && (
          <div className={`flex items-center gap-1 text-xs font-mono ${trend >= 0 ? 'text-green-400' : 'text-crimson-400'}`}>
            {trend >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            {Math.abs(trend)}%
          </div>
        )}
      </div>
      <div>
        <p className="font-display font-bold text-2xl text-white">{value}</p>
        <p className="text-xs text-slate-400 mt-0.5">{label}</p>
        {subtext && <p className="text-xs text-slate-500 mt-1">{subtext}</p>}
      </div>
    </motion.div>
  )
}

function DensityBadge({ cls }) {
  if (cls === 'high') return <span className="badge-high">HIGH</span>
  if (cls === 'medium') return <span className="badge-medium">MEDIUM</span>
  return <span className="badge-low">LOW</span>
}

const PIE_COLORS = ['#3B82F6', '#F59E0B', '#8B5CF6', '#22C55E']

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-white/10 rounded-xl px-3 py-2 text-xs">
      <p className="text-slate-400 mb-1">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }} className="font-mono">
          {p.name}: {p.value}
        </p>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [hourly, setHourly] = useState([])
  const [crossroad, setCrossroad] = useState(null)
  const [breakdown, setBreakdown] = useState([])
  const [trends, setTrends] = useState([])
  const [loading, setLoading] = useState(true)

  const fetchAll = useCallback(async () => {
    try {
      const [statsRes, hourlyRes, crossroadRes, breakdownRes, trendsRes] = await Promise.all([
        trafficAPI.getStats(),
        trafficAPI.getHourly(),
        crossroadAPI.getState(),
        analyticsAPI.getVehicleBreakdown(),
        analyticsAPI.getWeeklyTrends(),
      ])
      setStats(statsRes.data)
      setHourly(hourlyRes.data.hourly || [])
      setCrossroad(crossroadRes.data)
      setBreakdown(breakdownRes.data.breakdown || [])
      setTrends(trendsRes.data.trends || [])
    } catch (err) {
      console.error('Dashboard fetch error:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchAll()
    const interval = setInterval(fetchAll, 15000) // refresh every 15s
    return () => clearInterval(interval)
  }, [fetchAll])

  if (loading) return (
    <div className="flex items-center justify-center h-full">
      <div className="w-8 h-8 border-2 border-crimson-500/30 border-t-crimson-500 rounded-full animate-spin" />
    </div>
  )

  return (
    <div className="p-6 space-y-6 max-w-[1600px] mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-display font-bold text-2xl text-white">AI Dashboard</h1>
          <p className="text-sm text-slate-500 mt-0.5">Real-time traffic intelligence overview</p>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500 font-mono">
          <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
          Live · refreshes every 15s
        </div>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <StatCard icon={Car} label="Total Vehicles" value={stats?.total_vehicles?.toLocaleString() || '—'} color="blue" trend={4.2} delay={0} />
        <StatCard icon={Activity} label="Today's Vehicles" value={stats?.today_vehicles?.toLocaleString() || '—'} color="green" delay={0.05} />
        <StatCard icon={Siren} label="Ambulance Events" value={stats?.ambulance_events || 0} color="red" delay={0.1} subtext="Last 7 days" />
        <StatCard icon={Signal} label="Active Signals" value={stats?.active_signals || 0} color="amber" delay={0.15} />
        <StatCard icon={AlertTriangle} label="Emergency Active" value={stats?.emergency_signals || 0} color="red" delay={0.2} />
        <StatCard icon={TrendingUp} label="Avg Congestion" value={((stats?.avg_congestion_score || 0) * 100).toFixed(0) + '%'} color="violet" delay={0.25} />
      </div>

      {/* Charts Row 1 */}
      <div className="grid lg:grid-cols-3 gap-5">
        {/* Hourly vehicle count */}
        <div className="lg:col-span-2 glass-card p-5">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h3 className="font-display font-semibold text-sm text-white">Vehicle Count — Last 24 Hours</h3>
              <p className="text-xs text-slate-500 mt-0.5">All intersections combined</p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={hourly} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#EF4444" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
              <XAxis dataKey="hour" tick={{ fill: '#64748b', fontSize: 11 }}
                tickFormatter={h => `${h}:00`} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="vehicles" stroke="#EF4444"
                strokeWidth={2} fill="url(#areaGrad)" name="Vehicles" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Vehicle type pie */}
        <div className="glass-card p-5">
          <h3 className="font-display font-semibold text-sm text-white mb-5">Vehicle Type Mix</h3>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie data={breakdown} cx="50%" cy="50%" innerRadius={55} outerRadius={80}
                dataKey="count" nameKey="type" paddingAngle={3}>
                {breakdown.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-2 gap-2 mt-2">
            {breakdown.map((item, i) => (
              <div key={item.type} className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ backgroundColor: PIE_COLORS[i % PIE_COLORS.length] }} />
                <span className="text-xs text-slate-400 capitalize">{item.type}</span>
                <span className="text-xs font-mono text-white ml-auto">{item.count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="grid lg:grid-cols-2 gap-5">
        {/* Weekly trend */}
        <div className="glass-card p-5">
          <h3 className="font-display font-semibold text-sm text-white mb-5">Weekly Traffic Trend</h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={trends} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
              <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="total_vehicles" fill="#3B82F6" radius={[4, 4, 0, 0]} name="Vehicles" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Crossroad AI Signal Status */}
        <div className="glass-card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-display font-semibold text-sm text-white">Crossroad AI — Signal Status</h3>
            <div className="flex items-center gap-2">
              {crossroad?.current_phase && crossroad.signal_mode !== 'emergency' && (
                <span className="text-xs font-mono px-2 py-0.5 rounded-full border text-blue-400 bg-blue-500/10 border-blue-500/20">
                  Phase {crossroad.current_phase}
                </span>
              )}
              {crossroad && (
                <span className={`text-xs font-mono px-2 py-0.5 rounded-full border ${
                  crossroad.signal_mode === 'emergency'
                    ? 'text-red-400 bg-red-500/10 border-red-500/20'
                    : crossroad.signal_mode === 'auto'
                    ? 'text-green-400 bg-green-500/10 border-green-500/20'
                    : 'text-slate-400 bg-white/5 border-white/10'
                }`}>
                  {(crossroad.signal_mode || 'IDLE').toUpperCase()}
                </span>
              )}
            </div>
          </div>

          {(() => {
            const roads = crossroad?.roads || {}
            const active = Object.entries(roads).filter(([, s]) => s.status !== 'idle')
            if (active.length === 0) return (
              <div className="flex flex-col items-center justify-center py-6 text-center">
                <Signal className="w-8 h-8 text-slate-700 mb-2" />
                <p className="text-sm text-slate-500">No videos uploaded yet</p>
                <p className="text-xs text-slate-600 mt-1">Upload videos on Crossroad AI page to activate signal control</p>
              </div>
            )
            return (
              <div className="space-y-2">
                {['north','south','east','west'].map(road => {
                  const s = roads[road]
                  if (!s || s.status === 'idle') return null
                  const isGreen = s.signal === 'green'
                  return (
                    <div key={road} className={`flex items-center justify-between p-3 rounded-xl border transition-colors ${
                      isGreen ? 'bg-green-500/5 border-green-500/25' : 'bg-white/3 border-white/6'
                    }`}>
                      <div className="flex items-center gap-3">
                        <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${
                          isGreen ? 'bg-green-400 animate-pulse' : 'bg-red-500'
                        }`} />
                        <div>
                          <p className="text-sm font-medium text-white capitalize">{road} Road</p>
                          <p className="text-xs text-slate-500 font-mono">
                            {s.avg_vehicles ?? s.total_vehicles ?? 0} vehicles · PCU {(s.pcu_count || 0).toFixed(1)}
                          </p>
                          {(s.straight_count > 0 || s.right_count > 0) && (
                            <p className="text-xs text-slate-600 font-mono">
                              ↑{s.straight_count} straight · ↗{s.right_count} right
                            </p>
                          )}
                        </div>
                      </div>
                      <div className="text-right">
                        <p className={`text-sm font-mono font-bold ${isGreen ? 'text-green-400' : 'text-slate-500'}`}>
                          {isGreen ? `${s.green_duration}s GREEN` : 'RED'}
                        </p>
                        <p className="text-xs text-slate-600 font-mono capitalize">{s.density_class || '—'}</p>
                      </div>
                    </div>
                  )
                })}
                {crossroad.cycle_count > 0 && (
                  <p className="text-xs text-slate-600 font-mono pt-1">
                    {crossroad.cycle_count} AI signal cycle{crossroad.cycle_count !== 1 ? 's' : ''} completed
                  </p>
                )}
              </div>
            )
          })()}
        </div>
      </div>

      {/* Density distribution */}
      {stats?.density_distribution && (
        <div className="glass-card p-5">
          <h3 className="font-display font-semibold text-sm text-white mb-4">
            24h Density Distribution
          </h3>
          <div className="flex items-center gap-6">
            {Object.entries(stats.density_distribution).map(([cls, count]) => {
              const total = Object.values(stats.density_distribution).reduce((a, b) => a + b, 0)
              const pct = total ? ((count / total) * 100).toFixed(1) : 0
              const colors = { low: '#22C55E', medium: '#F59E0B', high: '#EF4444' }
              return (
                <div key={cls} className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-mono text-slate-400 uppercase">{cls}</span>
                    <span className="text-xs font-mono text-white">{pct}%</span>
                  </div>
                  <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${pct}%` }}
                      transition={{ duration: 1, ease: 'easeOut' }}
                      className="h-full rounded-full"
                      style={{ backgroundColor: colors[cls] }}
                    />
                  </div>
                  <p className="text-xs text-slate-600 mt-1">{count} events</p>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
