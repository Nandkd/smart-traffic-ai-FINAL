// frontend/src/components/layout/AppLayout.jsx — CROSSROAD VERSION
import React, { useState } from 'react'
import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuthStore } from '@/store/authStore'
import {
  LayoutDashboard, BarChart3, Brain, Siren,
  TrafficCone, ShieldCheck, LogOut, Menu, X,
  Zap, Camera
} from 'lucide-react'

const navItems = [
  {
    to: '/dashboard',
    icon: LayoutDashboard,
    label: 'Dashboard',
    desc: 'Overview & KPIs',
  },
  {
    to: '/crossroad',
    icon: Camera,
    label: 'Crossroad AI',
    desc: '4-road YOLO detection',
    highlight: true,
  },
  {
    to: '/analytics',
    icon: BarChart3,
    label: 'Analytics',
    desc: 'Heatmaps & trends',
  },
  {
    to: '/predict',
    icon: Brain,
    label: 'Congestion AI',
    desc: 'ML prediction',
  },
  {
    to: '/ambulance',
    icon: Siren,
    label: 'Ambulance AI',
    desc: 'CNN classifier',
  },
  {
    to: '/signals',
    icon: TrafficCone,
    label: 'Signal Control',
    desc: 'Manual timing',
  },
  {
    to: '/admin',
    icon: ShieldCheck,
    label: 'Admin',
    desc: 'System info',
    adminOnly: true,
  },
]

export default function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()

  const handleLogout = () => { logout(); navigate('/login') }

  const visibleItems = navItems.filter(
    item => !item.adminOnly || user?.role === 'admin'
  )

  return (
    <div className="flex h-screen bg-slate-950 overflow-hidden">

      <AnimatePresence initial={false}>
        {sidebarOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 248, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            className="flex flex-col border-r border-white/8
                       bg-slate-900/60 backdrop-blur-xl overflow-hidden flex-shrink-0"
          >
            {/* Logo */}
            <div className="flex items-center gap-3 px-5 py-5 border-b border-white/8">
              <div className="w-8 h-8 rounded-xl bg-crimson-600 flex items-center
                              justify-center shadow-lg shadow-crimson-600/30 flex-shrink-0">
                <Zap className="w-4 h-4 text-white" />
              </div>
              <div>
                <p className="font-display font-bold text-sm text-white">TrafficAI</p>
                <p className="text-xs text-slate-500">Indian Crossroad</p>
              </div>
            </div>

            {/* Nav */}
            <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
              {visibleItems.map(({ to, icon: Icon, label, desc, highlight }) => (
                <NavLink key={to} to={to}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3 py-2.5 rounded-xl
                     transition-all duration-150 group border ${
                      isActive
                        ? 'bg-crimson-600/15 border-crimson-600/25 text-crimson-400'
                        : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-white/5'
                    }`
                  }
                >
                  <Icon className="w-4 h-4 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium leading-tight truncate">{label}</p>
                    <p className="text-xs text-slate-600 group-hover:text-slate-500
                                  transition-colors truncate mt-0.5">{desc}</p>
                  </div>
                  {highlight && (
                    <span className="text-xs bg-crimson-600 text-white px-1.5 py-0.5
                                     rounded-full font-mono flex-shrink-0">AI</span>
                  )}
                </NavLink>
              ))}
            </nav>

            {/* User */}
            <div className="border-t border-white/8 p-3">
              <div className="flex items-center gap-3 px-3 py-2 rounded-xl
                              hover:bg-white/3 transition-colors group">
                <div className="w-7 h-7 rounded-full bg-gradient-to-br from-crimson-500
                                to-slate-700 flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-bold text-white uppercase">
                    {user?.username?.[0] || 'U'}
                  </span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-white truncate">{user?.username}</p>
                  <p className="text-xs text-slate-500 capitalize">{user?.role}</p>
                </div>
                <button onClick={handleLogout}
                  className="p-1.5 rounded-lg text-slate-500 hover:text-crimson-400
                             hover:bg-crimson-600/10 transition-colors opacity-0
                             group-hover:opacity-100">
                  <LogOut className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="flex items-center gap-4 px-6 py-3.5 border-b border-white/8
                           bg-slate-900/40 backdrop-blur-sm flex-shrink-0">
          <button onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg text-slate-400 hover:text-white
                       hover:bg-white/8 transition-colors">
            {sidebarOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
          </button>
          <div className="flex-1" />
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full
                               rounded-full bg-green-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
            </span>
            <span className="text-xs text-slate-400 font-mono">SYSTEM LIVE</span>
          </div>
        </header>
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
