"use client"

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'

const STORAGE_KEY = 'studyagent_auth'

export function useAuth(options = {}) {
  const { requireAuth = false, redirectTo = '/login' } = options
  const router = useRouter()
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    try {
      const raw = typeof window !== 'undefined' ? window.localStorage.getItem(STORAGE_KEY) : null
      if (raw) {
        const parsed = JSON.parse(raw)
        if (parsed && parsed.access_token && parsed.user) {
          setToken(parsed.access_token)
          setUser(parsed.user)
        }
      }
    } catch (e) {
      console.warn('useAuth_localStorage_read_failed', e)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!loading && requireAuth) {
      if (!token) {
        router.push(redirectTo)
      }
    }
  }, [loading, requireAuth, token, redirectTo, router])

  function saveAuth(nextUser, nextToken) {
    setUser(nextUser)
    setToken(nextToken)
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({ user: nextUser, access_token: nextToken }),
        )
      }
    } catch (e) {
      console.warn('useAuth_localStorage_write_failed', e)
    }
  }

  function logout() {
    setUser(null)
    setToken(null)
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.removeItem(STORAGE_KEY)
      }
    } catch (e) {
      console.warn('useAuth_localStorage_clear_failed', e)
    }
    router.push('/login')
  }

  return { user, token, loading, saveAuth, logout }
}
