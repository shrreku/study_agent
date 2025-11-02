"use client"
import { useState, useEffect } from 'react'

export default function AgentsPage() {
  const [sharedUserId, setSharedUserId] = useState('')
  const [userSavedMsg, setUserSavedMsg] = useState('')
  const [spConcepts, setSpConcepts] = useState('Derivatives, Integrals')
  const [spDailyMinutes, setSpDailyMinutes] = useState(60)
  const [spExamDate, setSpExamDate] = useState('')
  const [spResourceId, setSpResourceId] = useState('')
  const [studyPlan, setStudyPlan] = useState(null)
  const [spLoading, setSpLoading] = useState(false)

  const [dqConcepts, setDqConcepts] = useState('Heat Transfer, Conduction')
  const [dqCount, setDqCount] = useState(3)
  const [dqResourceId, setDqResourceId] = useState('')
  const [dailyQuiz, setDailyQuiz] = useState(null)
  const [dqLoading, setDqLoading] = useState(false)

  const [doubtQuestion, setDoubtQuestion] = useState("Explain Fourier's Law")
  const [doubtResourceId, setDoubtResourceId] = useState('')
  const [doubt, setDoubt] = useState(null)
  const [doubtLoading, setDoubtLoading] = useState(false)

  useEffect(() => {
    try {
      const cached = window.localStorage.getItem('studyagent_test_user_id')
      if (cached) {
        setSharedUserId(cached)
      }
    } catch (e) {
      console.warn('userId_localStorage_read_failed', e)
    }
  }, [])

  function saveSharedUserId() {
    setUserSavedMsg('')
    const trimmed = sharedUserId.trim()
    if (!trimmed) {
      setUserSavedMsg('Enter a UUID before saving')
      return
    }
    try {
      window.localStorage.setItem('studyagent_test_user_id', trimmed)
      setSharedUserId(trimmed)
      setUserSavedMsg('Saved! This ID will prefill compatible pages.')
    } catch (e) {
      setUserSavedMsg('Unable to access localStorage')
      console.warn('userId_localStorage_write_failed', e)
    }
  }

  async function callStudyPlan() {
    setSpLoading(true)
    setStudyPlan(null)
    try {
      const target_concepts = spConcepts.split(',').map(s => s.trim()).filter(Boolean)
      const body = { target_concepts, daily_minutes: Number(spDailyMinutes) }
      if (spExamDate) body.exam_date = spExamDate
      if (spResourceId) body.resource_id = spResourceId
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
      const body = { concepts, count: Number(dqCount) }
      if (dqResourceId) body.resource_id = dqResourceId
      const res = await fetch('http://localhost:8000/api/agent/daily-quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify(body),
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
      const body = { question: doubtQuestion }
      if (doubtResourceId) body.resource_id = doubtResourceId
      const res = await fetch('http://localhost:8000/api/agent/doubt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify(body),
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

      <section style={{ marginTop: 12, padding: 12, background: '#f7f7ff', border: '1px solid #e0e0ff' }}>
        <h2 style={{ marginTop: 0 }}>Test User ID</h2>
        <p style={{ marginBottom: 8 }}>
          Use a stable UUID here (for example the value of <code>TEST_USER_ID</code> in your <code>.env</code>). This value is cached locally and used by the Daily Quiz and Analysis pages when recording mastery.
        </p>
        <label>User ID: <input type="text" value={sharedUserId} onChange={e => setSharedUserId(e.target.value)} style={{ width: 360 }} placeholder="11111111-2222-3333-4444-555555555555" /></label>
        <button onClick={saveSharedUserId} style={{ marginLeft: 8 }}>Save for this browser</button>
        {userSavedMsg && <span style={{ marginLeft: 8, color: '#555' }}>{userSavedMsg}</span>}
      </section>

      <section style={{ marginTop: 16 }}>
        <h2>Study Plan</h2>
        <div style={{ marginBottom: 8 }}>
          <label>Target concepts: <input type="text" value={spConcepts} onChange={e => setSpConcepts(e.target.value)} style={{ width: 360 }} /></label>
        </div>
        <div style={{ marginBottom: 8 }}>
          <label>Daily minutes: <input type="number" value={spDailyMinutes} onChange={e => setSpDailyMinutes(e.target.value)} style={{ width: 80 }} /></label>
          <label style={{ marginLeft: 8 }}>Exam date: <input type="date" value={spExamDate} onChange={e => setSpExamDate(e.target.value)} /></label>
          <label style={{ marginLeft: 8 }}>Resource ID: <input type="text" value={spResourceId} onChange={e => setSpResourceId(e.target.value)} placeholder="optional" style={{ width: 260 }} /></label>
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
          <label style={{ marginLeft: 8 }}>Resource ID: <input type="text" value={dqResourceId} onChange={e => setDqResourceId(e.target.value)} placeholder="optional" style={{ width: 260 }} /></label>
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
          <label style={{ marginLeft: 8 }}>Resource ID: <input type="text" value={doubtResourceId} onChange={e => setDoubtResourceId(e.target.value)} placeholder="optional" style={{ width: 260 }} /></label>
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
