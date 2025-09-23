"use client"
import { useState } from 'react'

export default function AgentsPage() {
  const [spConcepts, setSpConcepts] = useState('Derivatives, Integrals')
  const [spDailyMinutes, setSpDailyMinutes] = useState(60)
  const [spExamDate, setSpExamDate] = useState('')
  const [studyPlan, setStudyPlan] = useState(null)
  const [spLoading, setSpLoading] = useState(false)

  const [dqConcepts, setDqConcepts] = useState('Heat Transfer, Conduction')
  const [dqCount, setDqCount] = useState(3)
  const [dailyQuiz, setDailyQuiz] = useState(null)
  const [dqLoading, setDqLoading] = useState(false)

  const [doubtQuestion, setDoubtQuestion] = useState("Explain Fourier's Law")
  const [doubt, setDoubt] = useState(null)
  const [doubtLoading, setDoubtLoading] = useState(false)

  async function callStudyPlan() {
    setSpLoading(true)
    setStudyPlan(null)
    try {
      const target_concepts = spConcepts.split(',').map(s => s.trim()).filter(Boolean)
      const body = { target_concepts, daily_minutes: Number(spDailyMinutes) }
      if (spExamDate) body.exam_date = spExamDate
      const res = await fetch('http://localhost:8000/api/agent/study-plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const j = await res.json()
      setStudyPlan(j)
    } catch (e) {
      setStudyPlan({ error: String(e) })
    } finally {
      setSpLoading(false)
    }
  }

  async function callDailyQuiz() {
    setDqLoading(true)
    setDailyQuiz(null)
    try {
      const concepts = dqConcepts.split(',').map(s => s.trim()).filter(Boolean)
      const res = await fetch('http://localhost:8000/api/agent/daily-quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify({ concepts, count: Number(dqCount) }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const j = await res.json()
      setDailyQuiz(j)
    } catch (e) {
      setDailyQuiz({ error: String(e) })
    } finally {
      setDqLoading(false)
    }
  }

  async function callDoubt() {
    setDoubtLoading(true)
    setDoubt(null)
    try {
      const res = await fetch('http://localhost:8000/api/agent/doubt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify({ question: doubtQuestion }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const j = await res.json()
      setDoubt(j)
    } catch (e) {
      setDoubt({ error: String(e) })
    } finally {
      setDoubtLoading(false)
    }
  }

  return (
    <main style={{ padding: 24 }}>
      <h1>Agents</h1>

      <section style={{ marginTop: 16 }}>
        <h2>Study Plan</h2>
        <div style={{ marginBottom: 8 }}>
          <label>Target concepts: <input type="text" value={spConcepts} onChange={e => setSpConcepts(e.target.value)} style={{ width: 360 }} /></label>
        </div>
        <div style={{ marginBottom: 8 }}>
          <label>Daily minutes: <input type="number" value={spDailyMinutes} onChange={e => setSpDailyMinutes(e.target.value)} style={{ width: 80 }} /></label>
          <label style={{ marginLeft: 8 }}>Exam date: <input type="date" value={spExamDate} onChange={e => setSpExamDate(e.target.value)} /></label>
        </div>
        <button onClick={callStudyPlan} disabled={spLoading}>Generate</button>
        {spLoading && <div style={{ marginTop: 8 }}>Generating...</div>}
        {studyPlan && (
          <div style={{ marginTop: 12 }}>
            {Array.isArray(studyPlan?.todos) ? (
              <ul>
                {studyPlan.todos.map((t, i) => (
                  <li key={i}>
                    {t.date} — {t.minutes || t.time_minutes || 30} min — {t.concept || t.title}
                    {Array.isArray(t.chunk_refs) && t.chunk_refs.length > 0 && (
                      <ul>
                        {t.chunk_refs.slice(0, 3).map((r, j) => (<li key={j} style={{ fontSize: 12, color: '#555' }}>{r.snippet}</li>))}
                      </ul>
                    )}
                  </li>
                ))}
              </ul>
            ) : (
              <pre style={{ background: '#f6f6f6', padding: 12 }}>{JSON.stringify(studyPlan, null, 2)}</pre>
            )}
          </div>
        )}
      </section>

      <section style={{ marginTop: 24 }}>
        <h2>Daily Quiz</h2>
        <div style={{ marginBottom: 8 }}>
          <label>Concepts: <input type="text" value={dqConcepts} onChange={e => setDqConcepts(e.target.value)} style={{ width: 360 }} /></label>
          <label style={{ marginLeft: 8 }}>Count: <input type="number" value={dqCount} onChange={e => setDqCount(e.target.value)} style={{ width: 80 }} /></label>
        </div>
        <button onClick={callDailyQuiz} disabled={dqLoading}>Get Quiz</button>
        {dqLoading && <div style={{ marginTop: 8 }}>Generating...</div>}
        {dailyQuiz && (
          <div style={{ marginTop: 12 }}>
            {Array.isArray(dailyQuiz?.items || dailyQuiz?.quiz) ? (
              <ol>
                {(dailyQuiz.items || dailyQuiz.quiz).map((q, i) => (
                  <li key={i} style={{ marginBottom: 8 }}>
                    <div><strong>{q.question}</strong></div>
                    <div>{Array.isArray(q.options) ? q.options.map((opt, idx) => (<div key={idx}>{String.fromCharCode(65 + idx)}. {opt}</div>)) : null}</div>
                    <div style={{ color: 'green' }}>Answer: {typeof q.answer_index === 'number' ? String.fromCharCode(65 + q.answer_index) : (q.answer || '')}</div>
                  </li>
                ))}
              </ol>
            ) : (
              <pre style={{ background: '#f6f6f6', padding: 12 }}>{JSON.stringify(dailyQuiz, null, 2)}</pre>
            )}
          </div>
        )}
      </section>

      <section style={{ marginTop: 24 }}>
        <h2>Doubt</h2>
        <div style={{ marginBottom: 8 }}>
          <label>Question: <input type="text" value={doubtQuestion} onChange={e => setDoubtQuestion(e.target.value)} style={{ width: 480 }} /></label>
        </div>
        <button onClick={callDoubt} disabled={doubtLoading}>Ask</button>
        {doubtLoading && <div style={{ marginTop: 8 }}>Thinking...</div>}
        {doubt && (
          <div style={{ marginTop: 12 }}>
            <div style={{ background: '#fff', padding: 12 }}>{doubt.answer || JSON.stringify(doubt)}</div>
            {Array.isArray(doubt.citations) && doubt.citations.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <strong>Citations</strong>
                <ul>
                  {doubt.citations.map((c, i) => (<li key={i}>{c.chunk_id || c.id} — {c.page || ''} — <span style={{ color: '#666' }}>{(c.snippet || '').slice(0, 160)}</span></li>))}
                </ul>
              </div>
            )}
          </div>
        )}
      </section>
    </main>
  )
}
