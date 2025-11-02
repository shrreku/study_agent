"use client"
import { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'

export default function StudyPlanPage() {
  const [conceptsText, setConceptsText] = useState('Transient conduction, Lumped capacitance')
  const [dailyMinutes, setDailyMinutes] = useState(30)
  const [examDate, setExamDate] = useState('')
  const [resourceId, setResourceId] = useState('')
  const [loading, setLoading] = useState(false)
  const [plan, setPlan] = useState(null)
  const [error, setError] = useState(null)
  const searchParams = useSearchParams()

  useEffect(() => {
    // Prefill from `?concepts=...` if present
    const qp = searchParams?.get('concepts')
    if (qp && qp.trim()) {
      setConceptsText(qp)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function generate() {
    setError(null)
    setPlan(null)
    const concepts = conceptsText.split(',').map(s => s.trim()).filter(Boolean)
    if (concepts.length === 0) return
    try {
      setLoading(true)
      const payload = { target_concepts: concepts, daily_minutes: Number(dailyMinutes) }
      if (examDate) payload.exam_date = examDate
      if (resourceId) payload.resource_id = resourceId
      const res = await fetch('http://localhost:8000/api/agent/study-plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setPlan(data)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <main style={{ padding: 24 }}>
      <h1>Study Plan</h1>
      <div style={{ marginBottom: 12 }}>
        <label>Concepts: <input type="text" value={conceptsText} onChange={e => setConceptsText(e.target.value)} style={{ width: 360 }} /></label>
        <label style={{ marginLeft: 8 }}>Daily minutes: <input type="number" value={dailyMinutes} onChange={e => setDailyMinutes(e.target.value)} style={{ width: 100 }} /></label>
        <label style={{ marginLeft: 8 }}>Exam date: <input type="date" value={examDate} onChange={e => setExamDate(e.target.value)} /></label>
        <label style={{ marginLeft: 8 }}>Resource ID: <input type="text" value={resourceId} onChange={e => setResourceId(e.target.value)} placeholder="optional" style={{ width: 260 }} /></label>
        <button onClick={generate} style={{ marginLeft: 8 }} disabled={loading}>Generate</button>
      </div>
      {error && <div style={{ color: 'red' }}>{error}</div>}
      {loading && <div>Loading…</div>}

      {plan && Array.isArray(plan.todos) && (
        <div style={{ marginTop: 12 }}>
          <h3>Todos</h3>
          <ul>
            {plan.todos.map((t, i) => (
              <li key={i} style={{ marginBottom: 10 }}>
                <div><strong>{t.date}</strong> — {t.minutes} min — {t.concept}</div>
                {t.summary && <div style={{ color: '#555' }}>{t.summary}</div>}
                {Array.isArray(t.chunk_refs) && t.chunk_refs.length > 0 && (
                  <div style={{ fontSize: 12, color: '#555' }}>
                    <strong>References:</strong> {t.chunk_refs.map((r, j) => (
                      <span key={j} style={{ marginRight: 8 }}>
                        <code>{r.chunk_id}</code> — {(r.snippet || '').slice(0, 120)}
                      </span>
                    ))}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </main>
  )
}
