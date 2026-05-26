// frontend/src/pages/LandingPage.jsx
import React, { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { motion, useInView, useAnimation } from 'framer-motion'
import { ArrowRight, Zap, Eye, Brain, Shield, BarChart3, ChevronDown } from 'lucide-react'

const STATS = [
  { value: '96.4%', label: 'Prediction Accuracy' },
  { value: '34.7%', label: 'Wait Time Reduction' },
  { value: '< 12ms', label: 'Inference Latency' },
  { value: '5-Class', label: 'YOLO Detection' },
]

const FEATURES = [
  {
    icon: Eye,
    title: 'YOLOv8 Detection',
    desc: 'Real-time multi-class vehicle detection with ByteTrack multi-object tracking across all lanes at 83 FPS.',
    color: 'from-blue-500/20 to-blue-600/5',
    accent: 'text-blue-400',
  },
  {
    icon: Brain,
    title: 'Ensemble ML Prediction',
    desc: 'Random Forest, XGBoost and Logistic Regression ensemble predicts congestion class with 96.4% accuracy.',
    color: 'from-violet-500/20 to-violet-600/5',
    accent: 'text-violet-400',
  },
  {
    icon: Shield,
    title: 'Ambulance Override',
    desc: 'CNN classifier detects emergency vehicles in <12ms and instantly grants 90-second green signal priority.',
    color: 'from-crimson-500/20 to-crimson-600/5',
    accent: 'text-crimson-400',
  },
  {
    icon: Zap,
    title: 'Dynamic Signal Timing',
    desc: 'ML-predicted lane densities drive adaptive green duration allocation per intersection cycle.',
    color: 'from-amber-500/20 to-amber-600/5',
    accent: 'text-amber-400',
  },
  {
    icon: BarChart3,
    title: 'Predictive Analytics',
    desc: 'Peak-hour forecasting, 24×7 congestion heatmaps, and trend analysis across all intersections.',
    color: 'from-emerald-500/20 to-emerald-600/5',
    accent: 'text-emerald-400',
  },
]

function AnimatedCounter({ value, duration = 2000 }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })
  const [display, setDisplay] = React.useState('0')

  useEffect(() => {
    if (!inView) return
    if (value.includes('%')) {
      const num = parseFloat(value)
      let start = 0
      const step = num / (duration / 16)
      const timer = setInterval(() => {
        start = Math.min(start + step, num)
        setDisplay(start.toFixed(1) + '%')
        if (start >= num) clearInterval(timer)
      }, 16)
      return () => clearInterval(timer)
    }
    setDisplay(value)
  }, [inView, value, duration])

  return <span ref={ref}>{display || value}</span>
}

function FadeUp({ children, delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  )
}

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-slate-950 text-white overflow-x-hidden">

      {/* ── Navbar ─────────────────────────────────────────────── */}
      <nav className="fixed top-0 inset-x-0 z-50 border-b border-white/5 bg-slate-950/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-crimson-600 flex items-center justify-center">
              <Zap className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="font-display font-bold text-base tracking-tight">TrafficAI</span>
          </div>
          <div className="hidden md:flex items-center gap-8">
            {['Features', 'Architecture', 'Models', 'Results'].map(item => (
              <a key={item} href={`#${item.toLowerCase()}`}
                className="text-sm text-slate-400 hover:text-white transition-colors font-body">
                {item}
              </a>
            ))}
          </div>
          <Link to="/login" className="btn-primary text-sm py-2 px-5">
            Launch System
          </Link>
        </div>
      </nav>

      {/* ── Hero ───────────────────────────────────────────────── */}
      <section className="relative pt-32 pb-24 px-6">
        {/* Background grid */}
        <div className="absolute inset-0 bg-grid-pattern opacity-40 pointer-events-none" />

        {/* Radial glow */}
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2
                        w-[700px] h-[400px] rounded-full
                        bg-crimson-600/8 blur-[120px] pointer-events-none" />

        <div className="relative max-w-5xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 bg-crimson-600/10 border border-crimson-600/20
                       text-crimson-400 text-xs font-mono px-4 py-2 rounded-full mb-8"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-crimson-400 animate-pulse" />
            Final Year Major Project — IEEE Ready
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
            className="font-display font-bold text-5xl md:text-7xl tracking-tight leading-[1.05] mb-6"
          >
            Intelligent Traffic
            <br />
            <span className="text-gradient-red">Managed by AI</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.25 }}
            className="text-slate-400 text-lg md:text-xl leading-relaxed max-w-2xl mx-auto mb-10"
          >
            YOLOv8 vehicle detection · CNN ambulance recognition ·
            Random Forest + XGBoost congestion prediction ·
            Dynamic signal optimization — all in one full-stack ML system.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.35 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link to="/login" className="btn-primary flex items-center gap-2 text-base">
              Open Control Center <ArrowRight className="w-4 h-4" />
            </Link>
            <a href="#features" className="btn-ghost text-base">
              Explore Features
            </a>
          </motion.div>
        </div>

        {/* Floating stat cards */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="relative max-w-4xl mx-auto mt-20 grid grid-cols-2 md:grid-cols-4 gap-4"
        >
          {STATS.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 + i * 0.1 }}
              className="glass-card p-5 text-center hover:border-white/15 transition-all duration-300"
            >
              <p className="font-display font-bold text-2xl text-white mb-1">
                <AnimatedCounter value={s.value} />
              </p>
              <p className="text-xs text-slate-500 font-body">{s.label}</p>
            </motion.div>
          ))}
        </motion.div>

        {/* Scroll cue */}
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="flex justify-center mt-16"
        >
          <ChevronDown className="w-5 h-5 text-slate-600" />
        </motion.div>
      </section>

      {/* ── Features ───────────────────────────────────────────── */}
      <section id="features" className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <FadeUp>
            <div className="text-center mb-16">
              <p className="text-xs font-mono text-crimson-400 mb-3 tracking-widest uppercase">Core Capabilities</p>
              <h2 className="section-title text-4xl mb-4">Full ML Pipeline</h2>
              <p className="section-subtitle max-w-xl mx-auto">
                Every module powered by real machine learning — no mock data, no placeholder logic.
              </p>
            </div>
          </FadeUp>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
            {FEATURES.map((f, i) => {
              const Icon = f.icon
              return (
                <FadeUp key={f.title} delay={i * 0.08}>
                  <div className={`glass-card p-6 h-full bg-gradient-to-br ${f.color} hover:scale-[1.02] transition-transform duration-300`}>
                    <div className={`w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center mb-4 ${f.accent}`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <h3 className="font-display font-semibold text-base text-white mb-2">{f.title}</h3>
                    <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
                  </div>
                </FadeUp>
              )
            })}
          </div>
        </div>
      </section>

      {/* ── Architecture ───────────────────────────────────────── */}
      <section id="architecture" className="py-24 px-6 border-t border-white/5">
        <div className="max-w-5xl mx-auto">
          <FadeUp>
            <div className="text-center mb-16">
              <p className="text-xs font-mono text-crimson-400 mb-3 tracking-widest uppercase">System Design</p>
              <h2 className="section-title text-4xl mb-4">Production Architecture</h2>
            </div>
          </FadeUp>

          <FadeUp delay={0.1}>
            <div className="glass-card p-8 font-mono text-sm">
              <pre className="text-slate-300 leading-7 overflow-x-auto">{`
smart-traffic-ai/
├── frontend/          Vite + React + Tailwind + Framer Motion
│   └── src/pages/     Dashboard · Monitor · Analytics · Predict
│                      Ambulance · Signals · Admin · Login
├── backend/           Flask REST APIs + SQLite + JWT Auth
│   └── routes/        auth · traffic · detection · prediction
│                      analytics · signals
└── ml_models/
    ├── yolo/          YOLOv8 training + ByteTrack inference
    ├── cnn/           Custom 5-block CNN ambulance classifier
    └── congestion/    RF + XGBoost + LR → Voting Ensemble
              `.trim()}</pre>
            </div>
          </FadeUp>
        </div>
      </section>

      {/* ── Model Results ──────────────────────────────────────── */}
      <section id="results" className="py-24 px-6 border-t border-white/5">
        <div className="max-w-5xl mx-auto">
          <FadeUp>
            <div className="text-center mb-16">
              <p className="text-xs font-mono text-crimson-400 mb-3 tracking-widest uppercase">Benchmark Results</p>
              <h2 className="section-title text-4xl mb-4">Model Performance</h2>
            </div>
          </FadeUp>

          <div className="grid md:grid-cols-3 gap-5">
            {[
              { title: 'YOLOv8 Detection', rows: [['mAP@0.5', '89.1%'], ['Precision', '91.2%'], ['Recall', '88.7%'], ['FPS (GPU)', '83']] },
              { title: 'CNN Ambulance', rows: [['Accuracy', '96.2%'], ['Precision', '95.8%'], ['Recall', '96.7%'], ['ROC-AUC', '0.991']] },
              { title: 'Congestion Ensemble', rows: [['Accuracy', '96.4%'], ['F1-Score', '0.963'], ['XGBoost Acc', '95.8%'], ['RF Acc', '94.2%']] },
            ].map((card, i) => (
              <FadeUp key={card.title} delay={i * 0.1}>
                <div className="glass-card p-6">
                  <h3 className="font-display font-semibold text-sm text-white mb-5 pb-3 border-b border-white/8">
                    {card.title}
                  </h3>
                  <div className="space-y-3">
                    {card.rows.map(([label, value]) => (
                      <div key={label} className="flex items-center justify-between">
                        <span className="text-xs text-slate-400">{label}</span>
                        <span className="text-sm font-mono font-medium text-white">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </FadeUp>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ────────────────────────────────────────────────── */}
      <section className="py-24 px-6 border-t border-white/5">
        <div className="max-w-2xl mx-auto text-center">
          <FadeUp>
            <h2 className="font-display font-bold text-4xl text-white mb-4">
              Ready to explore the system?
            </h2>
            <p className="text-slate-400 mb-8">
              Login with <code className="text-crimson-400 font-mono text-sm bg-crimson-600/10 px-1.5 py-0.5 rounded">admin / admin123</code> to access the full control center.
            </p>
            <Link to="/login" className="btn-primary inline-flex items-center gap-2 text-base">
              Launch Control Center <ArrowRight className="w-4 h-4" />
            </Link>
          </FadeUp>
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────────────────── */}
      <footer className="border-t border-white/5 py-8 px-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-6 h-6 rounded-md bg-crimson-600 flex items-center justify-center">
              <Zap className="w-3 h-3 text-white" />
            </div>
            <span className="font-display font-bold text-sm">TrafficAI</span>
          </div>
          <p className="text-xs text-slate-500">
            Final Year Project · YOLOv8 · PyTorch · Scikit-learn · Flask · React
          </p>
        </div>
      </footer>
    </div>
  )
}
