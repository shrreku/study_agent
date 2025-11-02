"use client"
import { useState, useEffect } from 'react'
import Script from 'next/script'

export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [resourceId, setResourceId] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [error, setError] = useState(null)
  const [chunkResult, setChunkResult] = useState(null)
  const [chunks, setChunks] = useState(null)
  const [chunksLimit, setChunksLimit] = useState(25)
  const [chunksOffset, setChunksOffset] = useState(0)
  const [chunksLoading, setChunksLoading] = useState(false)

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

  async function checkJobStatus() {
    if (!jobId) return
    setError(null)
    try {
      const res = await fetch(`http://localhost:8000/api/jobs/${jobId}`, {
        headers: { 'Authorization': 'Bearer test-token' },
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const j = await res.json()
      setJobStatus(j)
    } catch (e) {
      setError(String(e))
    }
  }

  async function callAdminRecompute() {
    setAdminResult(null)
    setError(null)
    try {
      const res = await fetch('http://localhost:8000/api/admin/recompute-search-tsv', {
        method: 'POST',
        headers: { 'Authorization': 'Bearer test-token' },
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const j = await res.json()
      setAdminResult(j)
    } catch (e) {
      setError(String(e))
    }
  }

  async function callBench() {
    setBenchResult(null)
    setError(null)
    try {
      const queries = benchQueries.split(',').map(s => s.trim()).filter(Boolean)
      const body = {
        queries,
        k: Number(benchK),
        sim_weight: Number(benchSimWeight),
        bm25_weight: Number(benchBm25Weight),
        resource_boost: Number(benchResourceBoost),
        page_proximity_boost: Boolean(benchPageProx),
      }
      const res = await fetch('http://localhost:8000/api/bench/pk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const j = await res.json()
      setBenchResult(j)
    } catch (e) {
      setError(String(e))
    }
  }

  // Math rendering helpers
  function triggerTypeset() {
    if (typeof window === 'undefined') return
    try {
      if (mathEngine === 'mathjax' && window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise()
      } else if (mathEngine === 'katex' && window.renderMathInElement) {
        const el = document.getElementById('math-root') || document.body
        window.renderMathInElement(el, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true },
          ],
        })
      }
    } catch (_) {
      // no-op
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
  const [mathEngine, setMathEngine] = useState('none') // 'none' | 'mathjax' | 'katex'
  // Admin & bench polish
  const [jobStatus, setJobStatus] = useState(null)
  const [adminResult, setAdminResult] = useState(null)
  const [benchQueries, setBenchQueries] = useState('heat flux, boundary layer')
  const [benchK, setBenchK] = useState(5)
  const [benchSimWeight, setBenchSimWeight] = useState(0.7)
  const [benchBm25Weight, setBenchBm25Weight] = useState(0.3)
  const [benchResourceBoost, setBenchResourceBoost] = useState(1.0)
  const [benchPageProx, setBenchPageProx] = useState(false)
  const [benchResult, setBenchResult] = useState(null)
  const [reindexResult, setReindexResult] = useState(null)

  useEffect(() => {
    if (mathEngine === 'katex') {
      const id = 'katex-css'
      if (typeof document !== 'undefined' && !document.getElementById(id)) {
        const link = document.createElement('link')
        link.id = id
        link.rel = 'stylesheet'
        link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css'
        document.head.appendChild(link)
      }
    }
    if (mathEngine !== 'none') {
      // delay to allow scripts to load and DOM to update
      setTimeout(triggerTypeset, 0)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mathEngine, chunks, studyPlan, dailyQuiz, doubtAnswer])

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

  async function fetchChunks(limit = chunksLimit, offset = chunksOffset) {
    if (!resourceId) return
    setChunksLoading(true)
    setError(null)
    try {
      const res = await fetch(`http://localhost:8000/api/resources/${resourceId}/chunks?limit=${limit}&offset=${offset}`, {
        headers: { 'Authorization': 'Bearer test-token' },
      })
      if (!res.ok) throw new Error(`Chunks fetch failed: ${res.status}`)
      const j = await res.json()
      setChunks(j)
      setChunksLimit(j.limit || limit)
      setChunksOffset(j.offset || offset)
    } catch (e) {
      setError(String(e))
    } finally {
      setChunksLoading(false)
    }
  }

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
        body: JSON.stringify({ question: doubtQuestion, resource_id: resourceId }),
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
      {mathEngine === 'mathjax' && (
        <>
          <Script id="mathjax-config" strategy="afterInteractive">
            {`
              window.MathJax = {
                tex: {
                  inlineMath: [['$', '$'], ['\\\(', '\\\)']],
                  displayMath: [['$$','$$'], ['\\\[','\\\]']]
                },
                options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }
              };
            `}
          </Script>
          <Script id="mathjax" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" strategy="afterInteractive" onLoad={() => { if (window.MathJax && window.MathJax.typesetPromise) window.MathJax.typesetPromise() }} />
        </>
      )}
      {mathEngine === 'katex' && (
        <>
          <Script id="katex-js" src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" strategy="afterInteractive" />
          <Script id="katex-auto-render" src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" strategy="afterInteractive" onLoad={() => { const el = document.getElementById('math-root') || document.body; if (window.renderMathInElement && el) window.renderMathInElement(el, { delimiters: [ { left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }, { left: '\\(', right: '\\)', display: false }, { left: '\\[', right: '\\]', display: true } ] }) }} />
        </>
      )}
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
        <div id="math-root" style={{ marginTop: 12 }}>
          <div>Resource created: <code>{resourceId}</code></div>
          {jobId && (
            <div>
              <a href={`/jobs/${jobId}`}>View job status</a>
            </div>
          )}
          <div style={{ marginTop: 8 }}>
            <button onClick={createChunks} disabled={!resourceId}>Create chunks</button>
            <button onClick={() => fetchChunks()} disabled={!resourceId} style={{ marginLeft: 8 }}>Load chunks</button>
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
                <label>Math rendering:
                  <select value={mathEngine} onChange={(e) => setMathEngine(e.target.value)} style={{ marginLeft: 8 }}>
                    <option value="none">None</option>
                    <option value="mathjax">MathJax</option>
                    <option value="katex">KaTeX</option>
                  </select>
                </label>
                <button onClick={triggerTypeset} disabled={mathEngine === 'none'} style={{ marginLeft: 8 }}>Re-typeset</button>
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
          {chunksLoading && <div style={{ marginTop: 8 }}>Loading chunks...</div>}
          {chunks && Array.isArray(chunks.chunks) && (
            <div style={{ marginTop: 12 }}>
              <h3>Chunks (page)</h3>
              <div style={{ maxHeight: 320, overflow: 'auto', background: '#fff', padding: 8 }}>
                {chunks.chunks.map((c) => (
                  <div key={c.id} style={{ padding: 8, borderBottom: '1px solid #eee' }}>
                    <div style={{ fontSize: 12, color: '#666' }}>Page: {c.page_number} • Offset: {c.source_offset}</div>
                    <div style={{ marginTop: 6, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{(() => { const t = c.snippet || (c.full_text ? c.full_text : ''); return t && t.length > 240 ? t.slice(0, 240) + '…' : t; })()}</div>
                    <div style={{ fontSize: 12, color: '#666', marginTop: 6 }}>Tags: {Array.isArray(c.concepts) ? c.concepts.join(', ') : ''} {c.chunk_type ? ' • ' + c.chunk_type : ''}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 8 }}>
                <button onClick={() => fetchChunks(chunksLimit, Math.max(0, chunksOffset - chunksLimit))} disabled={chunksOffset === 0}>Prev</button>
                <button onClick={() => fetchChunks(chunksLimit, chunksOffset + chunksLimit)} style={{ marginLeft: 8 }}>Next</button>
              </div>
            </div>
          )}
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
              {Array.isArray(dailyQuiz?.quiz || dailyQuiz?.items) ? (
                <ol>
                  {(dailyQuiz.quiz || dailyQuiz.items).map((q, i) => (
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
          <div style={{ marginTop: 16 }}>
            <strong>Admin & Bench</strong>
            <div style={{ marginTop: 8 }}>
              <button onClick={callAdminRecompute}>Recompute search_tsv</button>
              <button onClick={checkJobStatus} disabled={!jobId} style={{ marginLeft: 8 }}>Check job status</button>
              <button onClick={callReindex} disabled={!resourceId} style={{ marginLeft: 8 }}>Reindex resource</button>
            </div>
            {jobStatus && (
              <div style={{ marginTop: 8 }}>
                <pre style={{ background: '#f6f6f6', padding: 12 }}>{JSON.stringify(jobStatus, null, 2)}</pre>
              </div>
            )}
            {adminResult && (
              <div style={{ marginTop: 8 }}>
                <pre style={{ background: '#f6f6f6', padding: 12 }}>{JSON.stringify(adminResult, null, 2)}</pre>
              </div>
            )}
            {reindexResult && (
              <div style={{ marginTop: 8 }}>
                <pre style={{ background: '#f6f6f6', padding: 12 }}>{JSON.stringify(reindexResult, null, 2)}</pre>
              </div>
            )}
            <div style={{ marginTop: 8 }}>
              <div style={{ marginBottom: 8 }}>
                <label>Queries: <input type="text" value={benchQueries} onChange={e => setBenchQueries(e.target.value)} style={{ width: 360 }} /></label>
              </div>
              <div style={{ marginBottom: 8 }}>
                <label>k: <input type="number" value={benchK} onChange={e => setBenchK(e.target.value)} style={{ width: 60 }} /></label>
                <label style={{ marginLeft: 8 }}>sim_weight: <input type="number" step="0.1" value={benchSimWeight} onChange={e => setBenchSimWeight(e.target.value)} style={{ width: 80 }} /></label>
                <label style={{ marginLeft: 8 }}>bm25_weight: <input type="number" step="0.1" value={benchBm25Weight} onChange={e => setBenchBm25Weight(e.target.value)} style={{ width: 80 }} /></label>
                <label style={{ marginLeft: 8 }}>resource_boost: <input type="number" step="0.1" value={benchResourceBoost} onChange={e => setBenchResourceBoost(e.target.value)} style={{ width: 80 }} /></label>
                <label style={{ marginLeft: 8 }}>page_proximity: <input type="checkbox" checked={benchPageProx} onChange={e => setBenchPageProx(e.target.checked)} /></label>
                <button onClick={callBench} style={{ marginLeft: 8 }}>Run bench</button>
              </div>
              {benchResult && (
                <div style={{ marginTop: 8 }}>
                  <pre style={{ background: '#f6f6f6', padding: 12 }}>{JSON.stringify(benchResult, null, 2)}</pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {error && <div style={{ color: 'red', marginTop: 12 }}>{error}</div>}
    </main>
  )
}


