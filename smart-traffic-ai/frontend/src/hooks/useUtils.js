// frontend/src/hooks/useUtils.js
/**
 * Miscellaneous utility hooks.
 */
import { useState, useEffect, useRef, useCallback } from 'react'

// ── useDebounce ────────────────────────────────────────────────
/**
 * Debounce a rapidly-changing value by `delay` ms.
 * Useful for search inputs before firing API calls.
 */
export function useDebounce(value, delay = 300) {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(id)
  }, [value, delay])
  return debounced
}

// ── usePrevious ────────────────────────────────────────────────
/** Returns the previous render's value of a variable. */
export function usePrevious(value) {
  const ref = useRef()
  useEffect(() => { ref.current = value })
  return ref.current
}

// ── useInterval ────────────────────────────────────────────────
/** Safe setInterval hook that clears on unmount and respects callback changes. */
export function useInterval(callback, delay) {
  const savedCb = useRef(callback)
  useEffect(() => { savedCb.current = callback }, [callback])
  useEffect(() => {
    if (delay === null) return
    const id = setInterval(() => savedCb.current(), delay)
    return () => clearInterval(id)
  }, [delay])
}

// ── useWindowSize ──────────────────────────────────────────────
export function useWindowSize() {
  const [size, setSize] = useState({ width: window.innerWidth, height: window.innerHeight })
  useEffect(() => {
    const handler = () => setSize({ width: window.innerWidth, height: window.innerHeight })
    window.addEventListener('resize', handler)
    return () => window.removeEventListener('resize', handler)
  }, [])
  return size
}

// ── useToggle ─────────────────────────────────────────────────
export function useToggle(initial = false) {
  const [value, setValue] = useState(initial)
  const toggle = useCallback(() => setValue(v => !v), [])
  return [value, toggle, setValue]
}

// ── useAsync ──────────────────────────────────────────────────
/**
 * Run an async function and track loading/error/data state.
 *
 * Usage:
 *   const { data, loading, error, run } = useAsync(myApiCall)
 *   useEffect(() => { run(param) }, [])
 */
export function useAsync(asyncFn) {
  const [state, setState] = useState({ data: null, loading: false, error: null })
  const run = useCallback(async (...args) => {
    setState(s => ({ ...s, loading: true, error: null }))
    try {
      const data = await asyncFn(...args)
      setState({ data, loading: false, error: null })
      return data
    } catch (err) {
      setState({ data: null, loading: false, error: err })
      throw err
    }
  }, [asyncFn])
  return { ...state, run }
}
