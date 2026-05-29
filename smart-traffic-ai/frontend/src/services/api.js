// frontend/src/services/api.js
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// Response interceptor — auto-logout on 401
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('traffic-auth')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// ── Traffic API ────────────────────────────────────────────────
export const trafficAPI = {
  getStats: () => api.get('/traffic/stats'),
  getDensity: () => api.get('/traffic/density'),
  getHistory: (params) => api.get('/traffic/history', { params }),
  getHourly: () => api.get('/traffic/hourly'),
  addRecord: (data) => api.post('/traffic/record', data),
}

// ── Detection API ──────────────────────────────────────────────
export const detectionAPI = {
  detectVehicles: (formData) => api.post('/detect/vehicles', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }),
  detectAmbulance: (formData) => api.post('/detect/ambulance', formData),
}

// ── Prediction API ─────────────────────────────────────────────
export const predictionAPI = {
  predictCongestion: (data) => api.post('/predict/congestion', data),
  getPeakHours: (params) => api.get('/predict/peak-hours', { params }),
  optimizeSignal: (data) => api.post('/predict/signal-timing', data),
}

// ── Analytics API ──────────────────────────────────────────────
export const analyticsAPI = {
  getHeatmap: (params) => api.get('/analytics/heatmap', { params }),
  getVehicleBreakdown: (params) => api.get('/analytics/vehicle-breakdown', { params }),
  getWeeklyTrends: () => api.get('/analytics/trends'),
  getCongestionHistory: (params) => api.get('/analytics/congestion-history', { params }),
  getSummary: () => api.get('/analytics/summary'),
}

// ── Signals API ────────────────────────────────────────────────
export const signalsAPI = {
  getAll: () => api.get('/signals/'),
  getOne: (id) => api.get(`/signals/${id}`),
  update: (id, data) => api.patch(`/signals/${id}/update`, data),
  emergency: (id, lane) => api.post(`/signals/${id}/emergency`, { lane }),
  reset: (id) => api.post(`/signals/${id}/reset`),
}

// ── Crossroad API ──────────────────────────────────────────────
export const crossroadAPI = {
  getState: () => api.get('/crossroad/state'),
}

// ── Auth API ───────────────────────────────────────────────────
export const authAPI = {
  me: () => api.get('/auth/me'),
  listUsers: () => api.get('/auth/users'),
}

export default api
