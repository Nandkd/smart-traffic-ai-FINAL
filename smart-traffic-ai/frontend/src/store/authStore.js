// frontend/src/store/authStore.js
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import api from '@/services/api'

export const useAuthStore = create(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      loading: false,
      error: null,

      login: async (username, password) => {
        set({ loading: true, error: null })
        try {
          const { data } = await api.post('/auth/login', { username, password })
          api.defaults.headers.common['Authorization'] = `Bearer ${data.token}`
          set({ token: data.token, user: data.user, loading: false })
          return true
        } catch (err) {
          set({ error: err.response?.data?.error || 'Login failed', loading: false })
          return false
        }
      },

      register: async (username, email, password) => {
        set({ loading: true, error: null })
        try {
          const { data } = await api.post('/auth/register', { username, email, password })
          api.defaults.headers.common['Authorization'] = `Bearer ${data.token}`
          set({ token: data.token, user: data.user, loading: false })
          return true
        } catch (err) {
          set({ error: err.response?.data?.error || 'Registration failed', loading: false })
          return false
        }
      },

      logout: () => {
        delete api.defaults.headers.common['Authorization']
        set({ token: null, user: null })
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: 'traffic-auth',
      partialize: (state) => ({ token: state.token, user: state.user }),
      onRehydrateStorage: () => (state) => {
        if (state?.token) {
          api.defaults.headers.common['Authorization'] = `Bearer ${state.token}`
        }
      },
    }
  )
)
