// frontend/src/hooks/useTrafficData.js
/**
 * useTrafficData
 * ==============
 * Polls backend traffic endpoints on an interval and returns
 * live stats, hourly data, and per-intersection density.
 *
 * Usage:
 *   const { stats, hourly, density, loading, error, refresh } = useTrafficData(15000)
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { trafficAPI } from '@/services/api'

export default function useTrafficData(intervalMs = 15000) {
  const [stats, setStats] = useState(null)
  const [hourly, setHourly] = useState([])
  const [density, setDensity] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const intervalRef = useRef(null)

  const fetch = useCallback(async () => {
    try {
      const [statsRes, hourlyRes, densityRes] = await Promise.all([
        trafficAPI.getStats(),
        trafficAPI.getHourly(),
        trafficAPI.getDensity(),
      ])
      setStats(statsRes.data)
      setHourly(hourlyRes.data.hourly || [])
      setDensity(densityRes.data.data || [])
      setError(null)
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to fetch traffic data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetch()
    intervalRef.current = setInterval(fetch, intervalMs)
    return () => clearInterval(intervalRef.current)
  }, [fetch, intervalMs])

  return { stats, hourly, density, loading, error, refresh: fetch }
}
