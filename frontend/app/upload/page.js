"use client"
import { useState } from 'react'

export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [resourceId, setResourceId] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [error, setError] = useState(null)
  const [chunkResult, setChunkResult] = useState(null)

  const MAX_BYTES = 100 * 1024 * 1024 // 100MB

  function onFileChange(e) {
    setError(null)
    const f = e.target.files[0]
    if (!f) return
    if (f.size > MAX_BYTES) {
      setError('File exceeds 100MB limit')
      setFile(null)
      return
    }
    setFile(f)
  }

  async function upload() {
    if (!file) return
    setProgress(0)
    setError(null)
    setResourceId(null)
    setJobId(null)
    setChunkResult(null)

    const form = new FormData()
    form.append('file', file)

    try {
      const xhr = new window.XMLHttpRequest()
      xhr.open('POST', 'http://localhost:8000/api/resources/upload')
      // MVP backend requires a Bearer token presence; static token is sufficient
      xhr.setRequestHeader('Authorization', 'Bearer test-token')

      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable) {
          setProgress(Math.round((ev.loaded / ev.total) * 100))
        }
      }

      xhr.onreadystatechange = () => {
        if (xhr.readyState === 4) {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const json = JSON.parse(xhr.responseText)
              setResourceId(json.resource_id || json.id || null)
              setJobId(json.job_id || null)
            } catch (e) {
              setError('Upload succeeded but response parse failed')
            }
          } else {
            setError(`Upload failed: ${xhr.status}`)
          }
        }
      }

      xhr.send(form)
    } catch (e) {
      setError(String(e))
    }
  }

  async function createChunks() {
    if (!resourceId) return
    setError(null)
    setChunkResult(null)
    try {
      const res = await fetch(`http://localhost:8000/api/resources/${resourceId}/chunk`, {
        method: 'POST',
        headers: { 'Authorization': 'Bearer test-token' },
      })
      if (!res.ok) {
        throw new Error(`Chunk request failed: ${res.status}`)
      }
      const j = await res.json()
      setChunkResult(j)
    } catch (e) {
      setError(String(e))
    }
  }

  /* Agent actions: Study Plan, Daily Quiz, Doubt Chat */
  const [agentLoading, setAgentLoading] = useState(false)
  const [studyPlan, setStudyPlan] = useState(null)
  const [dailyQuiz, setDailyQuiz] = useState(null)
  const [doubtAnswer, setDoubtAnswer] = useState(null)
  const [examDate, setExamDate] = useState('')
  const [horizonWeeks, setHorizonWeeks] = useState(2)
  const [dailyMinutes, setDailyMinutes] = useState(30)
  const [quizCount, setQuizCount] = useState(5)
  const [quizConcepts, setQuizConcepts] = useState('')
  const [doubtQuestion, setDoubtQuestion] = useState('Explain the main concepts in this resource')

  async function callStudyPlan() {
    if (!resourceId) return
    setAgentLoading(true)
    setStudyPlan(null)
    try {
      const payload = { resource_id: resourceId, horizon_weeks: Number(horizonWeeks), daily_minutes: Number(dailyMinutes) }
      if (examDate) payload.exam_date = examDate
      const res = await fetch('http://localhost:8000/api/agent/study-plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const j = await res.json()
      setStudyPlan(j)
    } catch (e) {
      setError(String(e))
    } finally {
      setAgentLoading(false)
    }
  }

  async function callDailyQuiz() {
    if (!resourceId) return
    setAgentLoading(true)
    setDailyQuiz(null)
    try {
      const concepts = quizConcepts.split(',').map((s) => s.trim()).filter(Boolean)
      const res = await fetch('http://localhost:8000/api/agent/daily-quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify({ resource_id: resourceId, count: Number(quizCount), concepts }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const j = await res.json()
      setDailyQuiz(j)
    } catch (e) {
      setError(String(e))
    } finally {
      setAgentLoading(false)
    }
  }

  async function callDoubtChat() {
    if (!resourceId) return
    setAgentLoading(true)
    setDoubtAnswer(null)
    try {
      const res = await fetch('http://localhost:8000/api/agent/doubt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify({ question_text: doubtQuestion, resource_id: resourceId }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const j = await res.json()
      setDoubtAnswer(j)
    } catch (e) {
      setError(String(e))
    } finally {
      setAgentLoading(false)
    }
  }

  return (
    <main style={{ padding: 24 }}>
      <h1>Upload Resource</h1>

      <input type="file" onChange={onFileChange} />
      {file && <div>Selected: {file.name} ({Math.round(file.size/1024)} KB)</div>}

      <div style={{ marginTop: 12 }}>
        <button onClick={upload} disabled={!file}>Upload</button>
      </div>

      <div style={{ marginTop: 12 }}>
        <progress value={progress} max={100} style={{ width: '100%' }} />
        <div>{progress}%</div>
      </div>

      {resourceId && (
        <div style={{ marginTop: 12 }}>
          <div>Resource created: <code>{resourceId}</code></div>
          {jobId && (
            <div>
              <a href={`/jobs/${jobId}`}>View job status</a>
            </div>
          )}
          <div style={{ marginTop: 8 }}>
            <button onClick={createChunks} disabled={!resourceId}>Create chunks</button>
          </div>
          <div style={{ marginTop: 12 }}>
            <strong>Agents:</strong>
            <div style={{ marginTop: 8 }}>
              <div style={{ marginBottom: 8 }}>
                <label>Exam date: <input type="date" value={examDate} onChange={(e) => setExamDate(e.target.value)} /></label>
                <label style={{ marginLeft: 8 }}>Horizon (weeks): <input type="number" value={horizonWeeks} onChange={(e) => setHorizonWeeks(e.target.value)} style={{ width: 60 }} /></label>
                <label style={{ marginLeft: 8 }}>Daily minutes: <input type="number" value={dailyMinutes} onChange={(e) => setDailyMinutes(e.target.value)} style={{ width: 60 }} /></label>
              </div>
              <div style={{ marginBottom: 8 }}>
                <button onClick={callStudyPlan} disabled={!resourceId || agentLoading}>Study Plan</button>
              </div>
              <div style={{ marginBottom: 8 }}>
                <label>Quiz concepts (comma separated): <input type="text" value={quizConcepts} onChange={(e) => setQuizConcepts(e.target.value)} style={{ width: 300 }} /></label>
                <label style={{ marginLeft: 8 }}>Count: <input type="number" value={quizCount} onChange={(e) => setQuizCount(e.target.value)} style={{ width: 60 }} /></label>
                <button onClick={callDailyQuiz} disabled={!resourceId || agentLoading} style={{ marginLeft: 8 }}>Daily Quiz</button>
              </div>
              <div>
                <label>Doubt question: <input type="text" value={doubtQuestion} onChange={(e) => setDoubtQuestion(e.target.value)} style={{ width: 400 }} /></label>
                <button onClick={callDoubtChat} disabled={!resourceId || agentLoading} style={{ marginLeft: 8 }}>Doubt Chat</button>
              </div>
            </div>
          </div>
          {agentLoading && <div style={{ marginTop: 8 }}>Agent running...</div>}
          {studyPlan && (
            <div style={{ marginTop: 12 }}>
              <h3>Study Plan</h3>
              {Array.isArray(studyPlan?.todos) ? (
                <ul>
                  {studyPlan.todos.map((t, i) => (
                    <li key={i}>{t.date} — {t.time_minutes} min — {t.concept || t.title || JSON.stringify(t)}</li>
                  ))}
                </ul>
              ) : (
                <pre style={{ background: '#f6f6f6', padding: 12 }}>{JSON.stringify(studyPlan, null, 2)}</pre>
              )}
            </div>
          )}
          {dailyQuiz && (
            <div style={{ marginTop: 12 }}>
              <h3>Daily Quiz</h3>
              {Array.isArray(dailyQuiz?.items) ? (
                <ol>
                  {dailyQuiz.items.map((q, i) => (
                    <li key={i} style={{ marginBottom: 8 }}>
                      <div><strong>{q.question}</strong></div>
                      <div>{q.options?.map((opt, idx) => (<div key={idx}>{String.fromCharCode(65 + idx)}. {opt}</div>))}</div>
                      <div style={{ color: 'green' }}>Answer: {typeof q.answer_index === 'number' ? String.fromCharCode(65 + q.answer_index) : q.answer}</div>
                      <div style={{ fontSize: 12, color: '#666' }}>Source: {q.source_chunk_id || q.chunk_id || ''}</div>
                    </li>
                  ))}
                </ol>
              ) : (
                <pre style={{ background: '#f6f6f6', padding: 12 }}>{JSON.stringify(dailyQuiz, null, 2)}</pre>
              )}
            </div>
          )}
          {doubtAnswer && (
            <div style={{ marginTop: 12 }}>
              <h3>Doubt Chat Answer</h3>
              <div style={{ background: '#fff', padding: 12 }}>{doubtAnswer.answer || doubtAnswer.text || JSON.stringify(doubtAnswer)}</div>
              {Array.isArray(doubtAnswer.citations) && (
                <div style={{ marginTop: 8 }}>
                  <strong>Citations</strong>
                  <ul>
                    {doubtAnswer.citations.map((c, i) => (<li key={i}>{c.chunk_id || c.id} — {c.page || c.page_or_slide || ''}</li>))}
                  </ul>
                </div>
              )}
            </div>
          )}
          {chunkResult && (
            <div style={{ marginTop: 8 }}>
              Chunks created: <strong>{chunkResult.chunks_created}</strong>
            </div>
          )}
        </div>
      )}

      {error && <div style={{ color: 'red', marginTop: 12 }}>{error}</div>}
    </main>
  )
}


