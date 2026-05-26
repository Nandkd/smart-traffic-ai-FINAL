// frontend/src/pages/LoginPage.jsx
import React, { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Zap, Eye, EyeOff, AlertCircle, ArrowRight } from 'lucide-react'
import { useAuthStore } from '@/store/authStore'

export default function LoginPage() {
  const [mode, setMode] = useState('login')      // 'login' | 'register'
  const [form, setForm] = useState({ username: '', email: '', password: '' })
  const [showPass, setShowPass] = useState(false)
  const { login, register, loading, error, clearError, token } = useAuthStore()
  const navigate = useNavigate()

  useEffect(() => { if (token) navigate('/dashboard') }, [token, navigate])
  useEffect(() => { clearError() }, [mode])

  const handleSubmit = async (e) => {
    e.preventDefault()
    let ok
    if (mode === 'login') {
      ok = await login(form.username, form.password)
    } else {
      ok = await register(form.username, form.email, form.password)
    }
    if (ok) navigate('/dashboard')
  }

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center px-4 relative overflow-hidden">
      {/* BG decoration */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                      w-[600px] h-[400px] rounded-full bg-crimson-600/6 blur-[100px] pointer-events-none" />
      <div className="absolute inset-0 bg-grid-pattern opacity-30 pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
        className="relative w-full max-w-md"
      >
        {/* Card */}
        <div className="glass-card p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <Link to="/" className="inline-flex items-center gap-2 mb-6">
              <div className="w-9 h-9 rounded-xl bg-crimson-600 flex items-center justify-center">
                <Zap className="w-4.5 h-4.5 text-white" />
              </div>
              <span className="font-display font-bold text-lg">TrafficAI</span>
            </Link>
            <h1 className="font-display font-bold text-2xl text-white mb-2">
              {mode === 'login' ? 'Welcome back' : 'Create account'}
            </h1>
            <p className="text-sm text-slate-400">
              {mode === 'login'
                ? 'Sign in to the traffic control center'
                : 'Register for system access'}
            </p>
          </div>

          {/* Demo credentials hint */}
          {mode === 'login' && (
            <div className="mb-5 p-3 rounded-xl bg-blue-500/8 border border-blue-500/15 text-xs text-blue-300 font-mono">
              Demo — admin&nbsp;/&nbsp;admin123
            </div>
          )}

          {/* Error */}
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="mb-4 flex items-center gap-2.5 p-3 rounded-xl bg-crimson-600/10
                         border border-crimson-600/20 text-crimson-400 text-sm"
            >
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </motion.div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Username</label>
              <input
                type="text"
                value={form.username}
                onChange={e => setForm({ ...form, username: e.target.value })}
                className="w-full bg-slate-800/60 border border-white/10 rounded-xl px-4 py-3
                           text-sm text-white placeholder-slate-500
                           focus:outline-none focus:border-crimson-500/50 focus:ring-1 focus:ring-crimson-500/20
                           transition-colors"
                placeholder="e.g. admin"
                required
              />
            </div>

            {mode === 'register' && (
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">Email</label>
                <input
                  type="email"
                  value={form.email}
                  onChange={e => setForm({ ...form, email: e.target.value })}
                  className="w-full bg-slate-800/60 border border-white/10 rounded-xl px-4 py-3
                             text-sm text-white placeholder-slate-500
                             focus:outline-none focus:border-crimson-500/50 focus:ring-1 focus:ring-crimson-500/20
                             transition-colors"
                  placeholder="you@example.com"
                  required
                />
              </div>
            )}

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Password</label>
              <div className="relative">
                <input
                  type={showPass ? 'text' : 'password'}
                  value={form.password}
                  onChange={e => setForm({ ...form, password: e.target.value })}
                  className="w-full bg-slate-800/60 border border-white/10 rounded-xl px-4 py-3 pr-11
                             text-sm text-white placeholder-slate-500
                             focus:outline-none focus:border-crimson-500/50 focus:ring-1 focus:ring-crimson-500/20
                             transition-colors"
                  placeholder="••••••••"
                  required
                  minLength={6}
                />
                <button
                  type="button"
                  onClick={() => setShowPass(!showPass)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
                >
                  {showPass ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full btn-primary flex items-center justify-center gap-2 mt-2 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {loading
                ? <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                : <>
                    {mode === 'login' ? 'Sign In' : 'Create Account'}
                    <ArrowRight className="w-4 h-4" />
                  </>
              }
            </button>
          </form>

          {/* Toggle mode */}
          <p className="text-center text-xs text-slate-500 mt-6">
            {mode === 'login' ? "Don't have an account? " : 'Already have an account? '}
            <button
              onClick={() => setMode(mode === 'login' ? 'register' : 'login')}
              className="text-crimson-400 hover:text-crimson-300 font-medium transition-colors"
            >
              {mode === 'login' ? 'Register' : 'Sign In'}
            </button>
          </p>
        </div>

        <p className="text-center text-xs text-slate-600 mt-6">
          <Link to="/" className="hover:text-slate-400 transition-colors">← Back to homepage</Link>
        </p>
      </motion.div>
    </div>
  )
}
