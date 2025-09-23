"use client"
import { useState } from 'react'

export default function QuizPage() {
  const [conceptsText, setConceptsText] = useState('heat flux, Biot number')
  const [count, setCount] = useState(3)
  const [loading, setLoading] = useState(false)
  const [quiz, setQuiz] = useState([])
  const [idx, setIdx] = useState(0)
  const [choice, setChoice] = useState(null)
  const [graded, setGraded] = useState(false)
  const [error, setError] = useState(null)

  const current = quiz[idx]
  const correct = graded && current && typeof current.answer_index === 'number' && Number(choice) === Number(current.answer_index)

  async function startQuiz() {
    setError(null)
    setQuiz([])
    setIdx(0)
    setChoice(null)
    setGraded(false)
    const concepts = conceptsText.split(',').map(s => s.trim()).filter(Boolean)
    if (concepts.length === 0) return
    try {
      setLoading(true)
      const res = await fetch('http://localhost:8000/api/agent/daily-quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify({ concepts, count: Number(count) }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setQuiz(Array.isArray(data?.quiz) ? data.quiz : [])
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  function grade() {
    setGraded(true)
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
                <button onClick={next} disabled={idx >= quiz.length - 1}>Next</button>
              </div>
            </div>
          )}
        </div>
      )}

      {!loading && quiz.length > 0 && (
        <div style={{ marginTop: 12 }}>Question {idx + 1} of {quiz.length}</div>
      )}
    </main>
  )
}
