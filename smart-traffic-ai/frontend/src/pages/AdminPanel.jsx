// frontend/src/pages/AdminPanel.jsx
import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { ShieldCheck, Users, Database, Cpu, Activity, RefreshCw } from 'lucide-react'
import { authAPI, trafficAPI, analyticsAPI } from '@/services/api'

export default function AdminPanel() {
  const [users, setUsers] = useState([])
  const [stats, setStats] = useState(null)
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      authAPI.listUsers(),
      trafficAPI.getStats(),
      analyticsAPI.getSummary(),
    ]).then(([u, s, su]) => {
      setUsers(u.data.users || [])
      setStats(s.data)
      setSummary(su.data)
    }).finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-full">
      <div className="w-8 h-8 border-2 border-crimson-500/30 border-t-crimson-500 rounded-full animate-spin" />
    </div>
  )

  return (
    <div className="p-6 space-y-6 max-w-[1200px] mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-crimson-600/15 border border-crimson-600/25 flex items-center justify-center">
          <ShieldCheck className="w-4.5 h-4.5 text-crimson-400" />
        </div>
        <div>
          <h1 className="font-display font-bold text-2xl text-white">Admin Panel</h1>
          <p className="text-sm text-slate-500">System management · Role-protected access</p>
        </div>
      </div>

      {/* System metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { icon: Users, label: 'Registered Users', value: users.length, color: 'text-blue-400' },
          { icon: Database, label: 'Total Records', value: (stats?.total_vehicles || 0).toLocaleString(), color: 'text-violet-400' },
          { icon: Activity, label: 'System Uptime', value: `${stats?.system_uptime_pct || 99.4}%`, color: 'text-green-400' },
          { icon: Cpu, label: 'Active Signals', value: stats?.active_signals || 0, color: 'text-amber-400' },
        ].map((item, i) => {
          const Icon = item.icon
          return (
            <motion.div key={item.label} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.07 }} className="stat-card">
              <Icon className={`w-5 h-5 ${item.color}`} />
              <div>
                <p className={`font-display font-bold text-2xl ${item.color}`}>{item.value}</p>
                <p className="text-xs text-slate-400 mt-0.5">{item.label}</p>
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* User management */}
      <div className="glass-card p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-display font-semibold text-sm text-white flex items-center gap-2">
            <Users className="w-4 h-4 text-slate-400" /> User Management
          </h3>
          <span className="text-xs text-slate-500 font-mono">{users.length} users</span>
        </div>
        <div className="space-y-2">
          {users.map(user => (
            <div key={user.id}
              className="flex items-center gap-4 p-3 bg-white/3 rounded-xl border border-white/6">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-crimson-500 to-slate-700
                              flex items-center justify-center flex-shrink-0">
                <span className="text-xs font-bold text-white uppercase">{user.username[0]}</span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">{user.username}</p>
                <p className="text-xs text-slate-500 truncate">{user.email}</p>
              </div>
              <div className="flex items-center gap-3 flex-shrink-0">
                <span className={`text-xs font-mono px-2 py-0.5 rounded-full border ${
                  user.role === 'admin'
                    ? 'text-crimson-400 bg-crimson-500/10 border-crimson-500/20'
                    : 'text-slate-400 bg-white/5 border-white/10'
                }`}>{user.role}</span>
                <span className="text-xs text-slate-600 font-mono hidden md:block">
                  {user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model info */}
      {summary?.model_accuracy && (
        <div className="glass-card p-5">
          <h3 className="font-display font-semibold text-sm text-white mb-4 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-slate-400" /> Deployed Models
          </h3>
          <div className="space-y-3">
            {[
              { name: 'YOLOv8s Traffic Detector', version: 'v8.2.18', metric: `mAP@0.5: ${(summary.model_accuracy.yolo_map50 * 100).toFixed(1)}%`, status: 'active', file: 'yolov8_traffic.pt' },
              { name: 'CNN Ambulance Classifier', version: 'v1.0', metric: `Acc: ${(summary.model_accuracy.cnn_accuracy * 100).toFixed(1)}%`, status: 'active', file: 'ambulance_cnn.pth' },
              { name: 'Congestion Voting Ensemble', version: 'v1.0', metric: `F1: ${summary.model_accuracy.congestion_f1.toFixed(3)}`, status: 'active', file: 'congestion_ensemble.pkl' },
            ].map(model => (
              <div key={model.name}
                className="flex items-center gap-4 p-3 bg-white/3 rounded-xl border border-white/6">
                <span className="w-2 h-2 rounded-full bg-green-400 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-white">{model.name}</p>
                  <p className="text-xs text-slate-500 font-mono">{model.file}</p>
                </div>
                <div className="text-right flex-shrink-0">
                  <p className="text-xs font-mono text-green-400">{model.metric}</p>
                  <p className="text-xs text-slate-600">{model.version}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* System info */}
      <div className="glass-card p-5">
        <h3 className="font-display font-semibold text-sm text-white mb-4">System Information</h3>
        <div className="grid md:grid-cols-3 gap-4 text-sm">
          {[
            ['Backend', 'Flask 3.0.3 + SQLite'],
            ['ML Framework', 'PyTorch + Scikit-learn + XGBoost'],
            ['Detection', 'Ultralytics YOLOv8'],
            ['Frontend', 'Vite 5 + React 18 + Tailwind CSS'],
            ['Auth', 'JWT (flask-jwt-extended)'],
            ['Tracking', 'ByteTrack (multi-object)'],
          ].map(([k, v]) => (
            <div key={k} className="p-3 bg-white/3 rounded-xl border border-white/6">
              <p className="text-xs text-slate-500 mb-1">{k}</p>
              <p className="text-white font-mono text-xs">{v}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
