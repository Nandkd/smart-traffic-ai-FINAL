// frontend/src/App.jsx — CROSSROAD VERSION
import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'

import LandingPage       from '@/pages/LandingPage'
import LoginPage         from '@/pages/LoginPage'
import Dashboard         from '@/pages/Dashboard'
import CrossroadMonitor  from '@/pages/CrossroadMonitor'  // ← NEW main page
import Analytics         from '@/pages/Analytics'
import CongestionPredict from '@/pages/CongestionPredict'
import AmbulanceDetect   from '@/pages/AmbulanceDetect'
import SignalControl     from '@/pages/SignalControl'
import AdminPanel        from '@/pages/AdminPanel'

import AppLayout from '@/components/layout/AppLayout'

function ProtectedRoute({ children }) {
  const { token } = useAuthStore()
  if (!token) return <Navigate to="/login" replace />
  return children
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"      element={<LandingPage />} />
        <Route path="/login" element={<LoginPage />} />

        <Route element={<ProtectedRoute><AppLayout /></ProtectedRoute>}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/crossroad" element={<CrossroadMonitor />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/predict"   element={<CongestionPredict />} />
          <Route path="/ambulance" element={<AmbulanceDetect />} />
          <Route path="/signals"   element={<SignalControl />} />
          <Route path="/admin"     element={<AdminPanel />} />
        </Route>

        {/* Redirect old routes */}
        <Route path="/monitor"  element={<Navigate to="/crossroad" replace />} />
        <Route path="/live"     element={<Navigate to="/crossroad" replace />} />
        <Route path="/video"    element={<Navigate to="/crossroad" replace />} />
        <Route path="*"         element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
