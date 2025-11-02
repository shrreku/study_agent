"use client"

import { useEffect, useMemo, useRef, useState } from 'react'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

function newId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function parseConcepts(input) {
  return (input || '')
    .split(',')
    .map((c) => c.trim())
    .filter(Boolean)
}

export default function TutorTestingDashboard() {
  const [userId, setUserId] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [resourceId, setResourceId] = useState('')
  const [targetConceptsInput, setTargetConceptsInput] = useState('')
  const [message, setMessage] = useState('What concept should I revise today?')
  const [turns, setTurns] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showRaw, setShowRaw] = useState(false)
  const endRef = useRef(null)

  useEffect(() => {
    try {
      const cachedUser = window.localStorage.getItem('tutor_user_id')
      const cachedSession = window.localStorage.getItem('tutor_session_id')
      const cachedResource = window.localStorage.getItem('tutor_resource_id')
      const cachedConcepts = window.localStorage.getItem('tutor_target_concepts')
      if (cachedUser) setUserId(cachedUser)
      if (cachedSession) setSessionId(cachedSession)
      if (cachedResource) setResourceId(cachedResource)
      if (cachedConcepts) setTargetConceptsInput(cachedConcepts)
    } catch (e) {
      console.warn('tutor_dashboard_localStorage_read_failed', e)
    }
  }, [])

  useEffect(() => {
    if (endRef.current) {
      endRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [turns])

  function persistLocal(key, value) {
    try {
      if (value) {
        window.localStorage.setItem(key, value)
      } else {
        window.localStorage.removeItem(key)
      }
    } catch (e) {
      console.warn('tutor_dashboard_localStorage_write_failed', key, e)
    }
  }

  async function sendMessage() {
    if (loading) return
    const trimmedMessage = message.trim()
    const trimmedUser = userId.trim()
    if (!trimmedMessage || !trimmedUser) {
      setError('Provide both a message and user ID before sending.')
      return
    }
    const payload = {
      message: trimmedMessage,
      user_id: trimmedUser,
    }
    const trimmedSession = sessionId.trim()
    const trimmedResource = resourceId.trim()
    const targetConcepts = parseConcepts(targetConceptsInput)
    if (trimmedSession) payload.session_id = trimmedSession
    if (trimmedResource) payload.resource_id = trimmedResource
    if (targetConcepts.length > 0) payload.target_concepts = targetConcepts

    setError(null)
    setMessage('')
    setTurns((prev) => [
      ...prev,
      {
        id: newId(),
        role: 'user',
        content: trimmedMessage,
      },
    ])

    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/agent/tutor`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer test-token',
        },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        let detail = ''
        try {
          const errorPayload = await res.json()
          if (errorPayload?.detail) {
            detail = Array.isArray(errorPayload.detail) ? errorPayload.detail.join(', ') : String(errorPayload.detail)
          }
        } catch (parseErr) {
          try {
            const text = await res.text()
            detail = text?.slice(0, 400) || ''
          } catch (e) {
            /* ignore */
          }
        }
        const errMsg = detail ? `Tutor agent error (${res.status}): ${detail}` : `Tutor agent HTTP ${res.status}`
        throw new Error(errMsg)
      }
      const data = await res.json()

      const tutorTurn = {
        id: newId(),
        role: 'tutor',
        content: data.response || '',
        meta: {
          actionType: data.action_type,
          confidence: data.confidence,
          concept: data.concept,
          level: data.level,
          learningPath: data.learning_path,
          coldStart: data.cold_start,
          intent: data.intent,
          affect: data.affect,
          classificationConfidence: data.classification_confidence,
          sourceChunks: data.source_chunk_ids || [],
          raw: data,
        },
      }

      setTurns((prev) => [...prev, tutorTurn])
      if (data.session_id) {
        setSessionId(data.session_id)
        persistLocal('tutor_session_id', data.session_id)
      }
      persistLocal('tutor_user_id', trimmedUser)
      persistLocal('tutor_resource_id', trimmedResource)
      persistLocal('tutor_target_concepts', targetConceptsInput)
    } catch (err) {
      console.error('tutor_agent_call_failed', err)
      setError(String(err))
      setTurns((prev) => [
        ...prev,
        {
          id: newId(),
          role: 'tutor',
          content: "Sorry, I couldn't respond right now.",
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!loading) sendMessage()
    }
  }

  function resetSession() {
    setSessionId('')
    setTurns([])
    persistLocal('tutor_session_id', '')
  }

  function clearAll() {
    setSessionId('')
    setUserId('')
    setResourceId('')
    setTargetConceptsInput('')
    setTurns([])
    try {
      window.localStorage.removeItem('tutor_session_id')
      window.localStorage.removeItem('tutor_user_id')
      window.localStorage.removeItem('tutor_resource_id')
      window.localStorage.removeItem('tutor_target_concepts')
    } catch (e) {
      console.warn('tutor_dashboard_localStorage_clear_failed', e)
    }
  }

  function downloadTranscript() {
    const transcript = {
      generated_at: new Date().toISOString(),
      user_id: userId,
      session_id: sessionId,
      resource_id: resourceId,
      target_concepts: parseConcepts(targetConceptsInput),
      turns: turns.map((t) => ({
        role: t.role,
        content: t.content,
        meta: t.meta,
      })),
    }
    const blob = new Blob([JSON.stringify(transcript, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `tutor-session-${sessionId || 'new'}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const latestTutorMeta = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      const t = turns[i]
      if (t.role === 'tutor' && t.meta) {
        return t.meta
      }
    }
    return null
  }, [turns])

  return (
    <main style={{ padding: 24 }}>
      <h1>Tutor Agent Testing Dashboard</h1>
      <p style={{ maxWidth: 760, color: '#555' }}>
        Interact with the Tutor agent, inspect policy decisions, and download transcripts for regression checks. Provide a user ID to enable mastery-aware behaviour.
      </p>

      <section style={{ marginTop: 16, display: 'flex', gap: 24, flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 420px', minWidth: 320 }}>
          <h2>Controls</h2>
          <div style={{ marginBottom: 12 }}>
            <label style={{ display: 'block', marginBottom: 6 }}>User ID *</label>
            <input value={userId} onChange={(e) => setUserId(e.target.value)} style={{ width: '100%' }} placeholder="required" />
          </div>
          <div style={{ marginBottom: 12 }}>
            <label style={{ display: 'block', marginBottom: 6 }}>Session ID</label>
            <input value={sessionId} onChange={(e) => setSessionId(e.target.value)} style={{ width: '100%' }} placeholder="autofills after first response" />
          </div>
          <div style={{ marginBottom: 12 }}>
            <label style={{ display: 'block', marginBottom: 6 }}>Resource ID</label>
            <input value={resourceId} onChange={(e) => setResourceId(e.target.value)} style={{ width: '100%' }} placeholder="optional" />
          </div>
          <div style={{ marginBottom: 12 }}>
            <label style={{ display: 'block', marginBottom: 6 }}>Target Concepts (comma separated)</label>
            <input value={targetConceptsInput} onChange={(e) => setTargetConceptsInput(e.target.value)} style={{ width: '100%' }} placeholder="e.g. conduction, convection" />
          </div>
          <div style={{ marginBottom: 12 }}>
            <label style={{ display: 'block', marginBottom: 6 }}>Message *</label>
            <textarea value={message} onChange={(e) => setMessage(e.target.value)} onKeyDown={handleKeyPress} rows={3} style={{ width: '100%' }} placeholder="Ask or respond to the tutor..." />
          </div>
          {error && <div style={{ color: 'red', marginBottom: 12 }}>{error}</div>}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button onClick={sendMessage} disabled={loading || !message.trim() || !userId.trim()}>Send Turn</button>
            <button onClick={resetSession} disabled={loading}>Reset Session</button>
            <button onClick={clearAll} disabled={loading}>Clear All</button>
            <button onClick={() => setShowRaw((prev) => !prev)}>{showRaw ? 'Hide Raw' : 'Show Raw'}</button>
            <button onClick={downloadTranscript} disabled={turns.length === 0}>Download Transcript</button>
          </div>
        </div>

        <div style={{ flex: '1 1 360px', minWidth: 280 }}>
          <h2>Session Diagnostics</h2>
          {latestTutorMeta ? (
            <div style={{ background: '#f6f8fa', border: '1px solid #dde3ea', padding: 12, borderRadius: 6 }}>
              <div><strong>Focus Concept:</strong> {latestTutorMeta.concept || 'n/a'}</div>
              <div><strong>Student Level:</strong> {latestTutorMeta.level || 'n/a'}</div>
              <div><strong>Action Type:</strong> {latestTutorMeta.actionType}</div>
              <div><strong>Confidence:</strong> {typeof latestTutorMeta.confidence === 'number' ? latestTutorMeta.confidence.toFixed(2) : 'n/a'}</div>
              <div><strong>Intent:</strong> {latestTutorMeta.intent || 'n/a'} ({latestTutorMeta.classificationConfidence != null ? latestTutorMeta.classificationConfidence.toFixed(2) : 'n/a'})</div>
              <div><strong>Affect:</strong> {latestTutorMeta.affect || 'n/a'}</div>
              <div><strong>Cold Start:</strong> {latestTutorMeta.coldStart ? 'Yes' : 'No'}</div>
              <div><strong>Learning Path:</strong> {Array.isArray(latestTutorMeta.learningPath) && latestTutorMeta.learningPath.length > 0 ? latestTutorMeta.learningPath.join(' → ') : 'n/a'}</div>
              <div><strong>Source Chunks:</strong> {Array.isArray(latestTutorMeta.sourceChunks) && latestTutorMeta.sourceChunks.length > 0 ? latestTutorMeta.sourceChunks.map((c) => <code key={c} style={{ marginRight: 6 }}>{c}</code>) : 'n/a'}</div>
            </div>
          ) : (
            <div style={{ color: '#666' }}>Send at least one message to populate diagnostics.</div>
          )}
        </div>
      </section>

      <section style={{ marginTop: 24 }}>
        <h2>Conversation</h2>
        <div style={{ border: '1px solid #ddd', borderRadius: 6, padding: 12, minHeight: 220, background: '#fff' }}>
          {turns.length === 0 && <div style={{ color: '#777' }}>No turns yet. Start the session above.</div>}
          {turns.map((turn) => (
            <div key={turn.id} style={{ marginBottom: 16, paddingBottom: 12, borderBottom: '1px solid #f0f0f0' }}>
              <div style={{ fontWeight: 'bold', color: turn.role === 'user' ? '#222' : '#0a6' }}>{turn.role === 'user' ? 'You' : 'Tutor'}</div>
              <div style={{ whiteSpace: 'pre-wrap', marginTop: 4 }}>{turn.content}</div>
              {turn.meta && (
                <div style={{ marginTop: 6, fontSize: 13, color: '#555' }}>
                  <div>Action: <strong>{turn.meta.actionType}</strong> • Confidence: {typeof turn.meta.confidence === 'number' ? turn.meta.confidence.toFixed(2) : 'n/a'}</div>
                  <div>Concept: {turn.meta.concept || 'n/a'} • Level: {turn.meta.level || 'n/a'}</div>
                  <div>Intent: {turn.meta.intent || 'n/a'} • Affect: {turn.meta.affect || 'n/a'}</div>
                  {Array.isArray(turn.meta.sourceChunks) && turn.meta.sourceChunks.length > 0 && (
                    <div>Chunks: {turn.meta.sourceChunks.map((c) => <code key={c} style={{ marginRight: 6 }}>{c}</code>)}</div>
                  )}
                  {showRaw && (
                    <details style={{ marginTop: 6 }}>
                      <summary>Raw JSON</summary>
                      <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', marginTop: 4 }}>{JSON.stringify(turn.meta.raw, null, 2)}</pre>
                    </details>
                  )}
                </div>
              )}
            </div>
          ))}
          {loading && <div>Sending to tutor…</div>}
          <div ref={endRef} />
        </div>
      </section>
    </main>
  )
}
