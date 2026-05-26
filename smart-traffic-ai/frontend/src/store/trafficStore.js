// frontend/src/store/trafficStore.js
/**
 * Global Zustand store for traffic data and signal state.
 * Keeps data available across page navigations without re-fetching.
 */
import { create } from 'zustand'
import { trafficAPI, signalsAPI, analyticsAPI } from '@/services/api'

export const useTrafficStore = create((set, get) => ({
  // ── State ──────────────────────────────────────────────────
  stats: null,
  hourly: [],
  density: [],
  signals: [],
  weeklyTrends: [],
  vehicleBreakdown: [],
  loading: false,
  lastFetched: null,

  // ── Actions ────────────────────────────────────────────────
  fetchAll: async () => {
    set({ loading: true })
    try {
      const [statsRes, hourlyRes, densityRes, signalsRes, trendsRes, bdRes] = await Promise.all([
        trafficAPI.getStats(),
        trafficAPI.getHourly(),
        trafficAPI.getDensity(),
        signalsAPI.getAll(),
        analyticsAPI.getWeeklyTrends(),
        analyticsAPI.getVehicleBreakdown(),
      ])
      set({
        stats: statsRes.data,
        hourly: hourlyRes.data.hourly || [],
        density: densityRes.data.data || [],
        signals: signalsRes.data.signals || [],
        weeklyTrends: trendsRes.data.trends || [],
        vehicleBreakdown: bdRes.data.breakdown || [],
        lastFetched: Date.now(),
      })
    } catch (err) {
      console.error('trafficStore.fetchAll:', err)
    } finally {
      set({ loading: false })
    }
  },

  fetchSignals: async () => {
    try {
      const { data } = await signalsAPI.getAll()
      set({ signals: data.signals || [] })
    } catch (err) {
      console.error('trafficStore.fetchSignals:', err)
    }
  },

  updateSignal: (updatedSignal) => {
    set(state => ({
      signals: state.signals.map(s =>
        s.id === updatedSignal.id ? updatedSignal : s
      ),
    }))
  },

  // ── Computed getters ───────────────────────────────────────

  /** Stale if data is older than `ms` milliseconds. */
  isStale: (ms = 30000) => {
    const { lastFetched } = get()
    if (!lastFetched) return true
    return Date.now() - lastFetched > ms
  },

  /** Active emergency signals. */
  emergencySignals: () => get().signals.filter(s => s.status === 'emergency'),
}))
