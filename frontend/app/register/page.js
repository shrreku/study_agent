"use client"

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../hooks/useAuth'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

export default function RegisterPage() {
  const router = useRouter()
  const { saveAuth } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [consent, setConsent] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function onSubmit(e) {
    e.preventDefault()
    setError('')
    if (!email.trim() || !password) {
      setError('Enter email and password')
      return
    }
    if (!consent) {
      setError('You must consent to data collection for research to continue')
      return
    }
    try {
      setLoading(true)
      const body = {
        email: email.trim(),
        password,
        display_name: displayName.trim() || null,
      }
      const res = await fetch(`${API_BASE}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const msg = await res.text()
        throw new Error(msg || `HTTP ${res.status}`)
      }
      const data = await res.json()
      if (!data?.access_token || !data?.user) {
        throw new Error('Invalid auth response')
      }
      saveAuth(data.user, data.access_token)
      router.push('/upload')
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <main style={{ padding: 24, maxWidth: 640 }}>
      <h1>Create Account</h1>
      <form onSubmit={onSubmit} style={{ maxWidth: 420 }}>
        <div style={{ marginBottom: 12 }}>
          <label>Email<br />
            <input
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              style={{ width: '100%' }}
            />
          </label>
        </div>
        <div style={{ marginBottom: 12 }}>
          <label>Password<br />
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              style={{ width: '100%' }}
            />
          </label>
          <div style={{ fontSize: 12, color: '#555' }}>Use at least 8 characters.</div>
        </div>
        <div style={{ marginBottom: 12 }}>
          <label>Display name (optional)<br />
            <input
              type="text"
              value={displayName}
              onChange={e => setDisplayName(e.target.value)}
              style={{ width: '100%' }}
            />
          </label>
        </div>
        <div style={{ marginBottom: 12, padding: 12, background: '#f7f7ff', border: '1px solid #ddd' }}>
          <p style={{ marginTop: 0, marginBottom: 8 }}><strong>Consent & privacy</strong></p>
          <p style={{ marginTop: 0, marginBottom: 8, fontSize: 13, color: '#444' }}>
            Interactions with StudyAgent (questions, responses, notes, and other activity) are logged and may be
            used to build research datasets and improve the system. Please do not enter sensitive personal data
            (such as government IDs, financial details, or confidential information) in your questions or notes.
          </p>
          <label style={{ fontSize: 13 }}>
            <input
              type="checkbox"
              checked={consent}
              onChange={e => setConsent(e.target.checked)}
              style={{ marginRight: 6 }}
            />
            I understand and consent to data collection for research.
          </label>
        </div>
        {error && <div style={{ color: 'red', marginBottom: 12 }}>{error}</div>}
        <button type="submit" disabled={loading}>{loading ? 'Creating account…' : 'Create account'}</button>
      </form>
      <div style={{ marginTop: 16 }}>
        Already have an account? <a href="/login">Log in</a>
      </div>
    </main>
  )
}
