// frontend/src/hooks/useSignalTimer.js
/**
 * useSignalTimer
 * ==============
 * Animates a countdown from `seconds` down to 0, then calls onComplete.
 * Used in the Signal Control page to show current green-phase countdown.
 *
 * Usage:
 *   const { remaining, phase, start, stop, reset } = useSignalTimer(30, 'north', onExpiry)
 */

import { useState, useEffect, useRef, useCallback } from 'react'

export default function useSignalTimer(initialSeconds = 30, lane = 'north', onComplete = null) {
  const [remaining, setRemaining] = useState(initialSeconds)
  const [running, setRunning] = useState(false)
  const [phase, setPhase] = useState(lane)
  const timerRef = useRef(null)

  const tick = useCallback(() => {
    setRemaining(prev => {
      if (prev <= 1) {
        clearInterval(timerRef.current)
        setRunning(false)
        onComplete?.()
        return 0
      }
      return prev - 1
    })
  }, [onComplete])

  const start = useCallback(() => {
    if (running) return
    setRunning(true)
    timerRef.current = setInterval(tick, 1000)
  }, [running, tick])

  const stop = useCallback(() => {
    clearInterval(timerRef.current)
    setRunning(false)
  }, [])

  const reset = useCallback((newSeconds = initialSeconds, newLane = lane) => {
    stop()
    setRemaining(newSeconds)
    setPhase(newLane)
  }, [stop, initialSeconds, lane])

  useEffect(() => () => clearInterval(timerRef.current), [])

  // Percentage of time elapsed
  const progress = 1 - remaining / (initialSeconds || 1)

  return { remaining, progress, running, phase, start, stop, reset }
}
