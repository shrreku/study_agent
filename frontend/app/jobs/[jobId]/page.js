"use client"
import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import { useAuth } from '../../hooks/useAuth'
import { API_BASE } from '../../lib/api'

export default function JobStatusPage() {
  const { token } = useAuth({ requireAuth: true })
  const params = useParams()
  const jobId = params?.jobId
  const [status, setStatus] = useState('loading')
  const [payload, setPayload] = useState(null)
  const [error, setError] = useState(null)

  function authHeader() {
    return token ? `Bearer ${token}` : 'Bearer test-token'
  }

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const res = await fetch(`${API_BASE}/api/jobs/${jobId}`, {
          headers: { 'Authorization': authHeader() },
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const j = await res.json()
        if (cancelled) return
        setStatus(j.status || 'unknown')
        setPayload(j)
      } catch (e) {
        if (!cancelled) setError(String(e))
      }
    }

    poll()
    const iv = setInterval(poll, 3000)
    return () => {
      cancelled = true
      clearInterval(iv)
    }
  }, [jobId])

  return (
    <main style={{ padding: 24 }}>
      <h1>Job Status: {jobId}</h1>
      {error && <div style={{ color: 'red' }}>{error}</div>}
      <div style={{ marginTop: 12 }}>
        <strong>Status:</strong> {status}
      </div>
      <div style={{ marginTop: 12 }}>
        <pre style={{ background: '#f6f6f6', padding: 12 }}>{JSON.stringify(payload, null, 2)}</pre>
      </div>
    </main>
  )
}


