"use client"
import { useState } from 'react'
import { useRouter } from 'next/navigation'

export default function AnalysisPage() {
  const [userId, setUserId] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [data, setData] = useState(null)
  const router = useRouter()

  async function runAnalysis() {
    setError(null)
    setData(null)
    if (!userId.trim()) {
      setError('Enter user_id')
      return
    }
    try {
      setLoading(true)
      const res = await fetch('http://localhost:8000/api/agent/analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify({ user_id: userId.trim() })
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const j = await res.json()
      setData(j)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  function startPlanFromNext() {
    const next = Array.isArray(data?.next_concepts) ? data.next_concepts.filter(Boolean) : []
    if (next.length === 0) return
    const q = encodeURIComponent(next.join(', '))
    router.push(`/agents/study-plan?concepts=${q}`)
  }

  return (
    <main style={{ padding: 24 }}>
      <h1>Analysis</h1>
      <div style={{ marginBottom: 12 }}>
        <label>User ID: <input type="text" value={userId} onChange={e => setUserId(e.target.value)} style={{ width: 360 }} /></label>
        <button onClick={runAnalysis} style={{ marginLeft: 8 }} disabled={loading}>Analyze</button>
      </div>
      {loading && <div>Loading…</div>}
      {error && <div style={{ color: 'red' }}>{error}</div>}

      {data && (
        <div style={{ marginTop: 12 }}>
          <h3>Summary</h3>
          <div style={{ background: '#fff', padding: 12 }}>{data.summary || ''}</div>

          <div style={{ display: 'flex', gap: 16, marginTop: 16 }}>
            <div style={{ flex: 1 }}>
              <h3>Strengths</h3>
              <table style={{ width: '100%', background: '#fff' }}>
                <thead>
                  <tr><th align="left">Concept</th><th align="left">Mastery</th><th align="left">Attempts</th><th align="left">Correct rate</th></tr>
                </thead>
                <tbody>
                  {(data.strengths || []).map((s, i) => (
                    <tr key={i}>
                      <td>{s.concept}</td>
                      <td>{s.mastery}</td>
                      <td>{s.attempts}</td>
                      <td>{s.correct_rate}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ flex: 1 }}>
              <h3>Weaknesses</h3>
              <table style={{ width: '100%', background: '#fff' }}>
                <thead>
                  <tr><th align="left">Concept</th><th align="left">Mastery</th><th align="left">Attempts</th><th align="left">Correct rate</th></tr>
                </thead>
                <tbody>
                  {(data.weaknesses || []).map((w, i) => (
                    <tr key={i}>
                      <td>{w.concept}</td>
                      <td>{w.mastery}</td>
                      <td>{w.attempts}</td>
                      <td>{w.correct_rate}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <h3 style={{ marginTop: 16 }}>References</h3>
          <div>
            {(data.references || []).map((r, i) => (
              <div key={i} style={{ marginBottom: 8 }}>
                <strong>{r.concept}</strong>
                <ul>
                  {(r.refs || []).map((ref, j) => (
                    <li key={j}><code>{ref.chunk_id}</code> — {(ref.snippet || '').slice(0, 160)}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          <div style={{ marginTop: 16 }}>
            <button onClick={startPlanFromNext} disabled={!Array.isArray(data.next_concepts) || data.next_concepts.length === 0}>Create study plan from next concepts</button>
          </div>
        </div>
      )}

      {!data && !loading && (
        <div style={{ marginTop: 12, color: '#555' }}>
          No data yet. Take a quiz or ask doubts to populate mastery and weaknesses signals.
        </div>
      )}
    </main>
  )
}
