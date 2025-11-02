"use client"
import { useState, useEffect } from 'react'

export default function QuizPage() {
  const [conceptsText, setConceptsText] = useState('heat flux, Biot number')
  const [count, setCount] = useState(3)
  const [resourceId, setResourceId] = useState('')
  const [userId, setUserId] = useState('')
  const [loading, setLoading] = useState(false)
  const [quiz, setQuiz] = useState([])
  const [idx, setIdx] = useState(0)
  const [choice, setChoice] = useState(null)
  const [graded, setGraded] = useState(false)
  const [error, setError] = useState(null)
  const [submitMsg, setSubmitMsg] = useState('')
  const [masteryUpdates, setMasteryUpdates] = useState([])

  const current = quiz[idx]
  const correct = graded && current && typeof current.answer_index === 'number' && Number(choice) === Number(current.answer_index)

  useEffect(() => {
    try {
      const cached = window.localStorage.getItem('studyagent_test_user_id')
      if (cached) {
        setUserId(cached)
      }
    } catch (e) {
      console.warn('quiz_user_id_localStorage_read_failed', e)
    }
  }, [])

  async function startQuiz() {
    setError(null)
    setQuiz([])
    setIdx(0)
    setChoice(null)
    setGraded(false)
    setMasteryUpdates([])
    const concepts = conceptsText.split(',').map(s => s.trim()).filter(Boolean)
    if (concepts.length === 0) return
    try {
      setLoading(true)
      const body = { concepts, count: Number(count) }
      if (resourceId) body.resource_id = resourceId
      const res = await fetch('http://localhost:8000/api/agent/daily-quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setQuiz(Array.isArray(data?.quiz) ? data.quiz : [])
    } catch (e) {
      setError(String(e))
      setLoading(false)
    }
  }

  function grade() {
    setGraded(true)
  }

  async function submitGrade() {
    setSubmitMsg('')
    if (!graded || current == null || choice == null) return
    const trimmed = userId.trim()
    if (!trimmed) {
      setSubmitMsg('Provide a user ID before recording')
      return
    }
    try {
      const body = {
        quiz_id: 'local-ui',
        answers: [
          {
            concept: current.concept || 'Unknown',
            chosen: Number(choice),
            correct_index: Number(current.answer_index),
          },
        ],
        user_id: trimmed,
      }
      const res = await fetch('http://localhost:8000/api/agent/quiz/answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const j = await res.json()
      try {
        window.localStorage.setItem('studyagent_test_user_id', trimmed)
      } catch (e) {
        console.warn('quiz_user_id_localStorage_write_failed', e)
      }
      setSubmitMsg(`Recorded (${j.graded || 0})`)
      setMasteryUpdates(Array.isArray(j?.updates) ? j.updates : [])
    } catch (e) {
      setSubmitMsg(`Failed: ${String(e)}`)
    }
  }

  function next() {
    setChoice(null)
    setGraded(false)
    setIdx(i => Math.min(i + 1, Math.max(0, quiz.length - 1)))
  }

  return (
    <main style={{ padding: 24 }}>
      <h1>Daily Quiz</h1>
      <div style={{ marginBottom: 12 }}>
        <label>Concepts: <input type="text" value={conceptsText} onChange={e => setConceptsText(e.target.value)} style={{ width: 360 }} /></label>
        <label style={{ marginLeft: 8 }}>Count: <input type="number" value={count} onChange={e => setCount(e.target.value)} style={{ width: 80 }} /></label>
        <label style={{ marginLeft: 8 }}>Resource ID: <input type="text" value={resourceId} onChange={e => setResourceId(e.target.value)} placeholder="optional" style={{ width: 260 }} /></label>
        <button onClick={startQuiz} style={{ marginLeft: 8 }} disabled={loading}>Start</button>
      </div>

      {loading && <div>Loading…</div>}
      {error && <div style={{ color: 'red' }}>{error}</div>}

      {current && (
        <div style={{ background: '#fff', padding: 16, border: '1px solid #eee', maxWidth: 720 }}>
          <div style={{ marginBottom: 8 }}><strong>Q{idx + 1}.</strong> {current.question}</div>
          <div style={{ marginBottom: 8 }}>
            {(current.options || []).map((opt, i) => (
              <div key={i}>
                <label>
                  <input type="radio" name="opt" checked={choice === i} onChange={() => setChoice(i)} disabled={graded} />{' '}
                  {String.fromCharCode(65 + i)}. {opt}
                </label>
              </div>
            ))}
          </div>
          {!graded && <button onClick={grade} disabled={choice === null}>Check answer</button>}
          {graded && (
            <div style={{ marginTop: 8 }}>
              <div style={{ color: correct ? 'green' : 'crimson' }}>
                {correct ? 'Correct!' : `Incorrect. Correct answer is ${typeof current.answer_index === 'number' ? String.fromCharCode(65 + current.answer_index) : '?'}`}
              </div>
              {current.explanation && (
                <div style={{ marginTop: 6 }}>
                  <strong>Explanation:</strong> {current.explanation}
                </div>
              )}
              {(current.references || []).length > 0 && (
                <div style={{ marginTop: 6 }}>
                  <strong>Reference:</strong> {(current.references || []).map((r, i) => (
                    <span key={i} style={{ marginRight: 8 }}>
                      <code>{r.chunk_id}</code> — {(r.snippet || '').slice(0, 120)}
                    </span>
                  ))}
                </div>
              )}
              <div style={{ marginTop: 8 }}>
                <button onClick={submitGrade}>Record result</button>
                {submitMsg && <span style={{ marginLeft: 8, color: '#555' }}>{submitMsg}</span>}
              </div>
              <div style={{ marginTop: 8 }}>
                <button onClick={next} disabled={idx >= quiz.length - 1}>Next</button>
              </div>
            </div>
          )}
        </div>
      )}

      {!loading && quiz.length > 0 && (
        <div style={{ marginTop: 12 }}>Question {idx + 1} of {quiz.length}</div>
      )}

      {masteryUpdates.length > 0 && (
        <section style={{ marginTop: 24, maxWidth: 720, border: '1px solid #ccd', background: '#f8fbff', padding: 12 }}>
          <h2 style={{ marginTop: 0, fontSize: 18 }}>Mastery impact</h2>
          <p style={{ marginBottom: 12 }}>The latest recording adjusted your mastery model per concept. Positive values push concepts toward strengths; negative values flag weaknesses.</p>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', borderBottom: '1px solid #ddd' }}>
                <th style={{ padding: '4px 8px' }}>Concept</th>
                <th style={{ padding: '4px 8px' }}>Outcome</th>
                <th style={{ padding: '4px 8px' }}>Δ mastery</th>
                <th style={{ padding: '4px 8px' }}>Mastery now</th>
                <th style={{ padding: '4px 8px' }}>Attempts</th>
              </tr>
            </thead>
            <tbody>
              {masteryUpdates.map((u, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '4px 8px' }}>{u.concept || 'Unknown'}</td>
                  <td style={{ padding: '4px 8px', color: u.correct ? 'green' : 'crimson' }}>{u.correct ? 'Correct' : 'Incorrect'}</td>
                  <td style={{ padding: '4px 8px', color: (u.delta || 0) >= 0 ? 'green' : 'crimson' }}>{(u.delta || 0).toFixed(3)}</td>
                  <td style={{ padding: '4px 8px' }}>{u.mastery != null ? u.mastery.toFixed(3) : '—'}</td>
                  <td style={{ padding: '4px 8px' }}>{u.attempts != null ? `${u.correct_attempts || 0}/${u.attempts}` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 12 }}>
            <a href="/agents/analysis" style={{ color: '#0052cc', textDecoration: 'underline' }}>Open Analysis agent</a> with the same user ID to see updated strengths & weaknesses.
          </div>
        </section>
      )}
    </main>
  )
}
