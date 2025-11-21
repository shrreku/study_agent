"use client"

import { useEffect, useMemo, useRef, useState } from 'react'
import { useAuth } from '../../hooks/useAuth'
import { API_BASE } from '../../lib/api'
import { loadNotesMetadata, mapJobStatusToLabel } from '../../lib/notes'
import ConceptsSidebar from '../ConceptsSidebar'

function newId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export default function PedagogicalTutorPage() {
  const { user, token } = useAuth({ requireAuth: true })

  const [notes, setNotes] = useState([])
  const [selectedNoteIds, setSelectedNoteIds] = useState([])

  const [concepts, setConcepts] = useState([])
  const [conceptsLoading, setConceptsLoading] = useState(false)
  const [conceptsError, setConceptsError] = useState(null)
  const [conceptFeedbackError, setConceptFeedbackError] = useState(null)
  const [selectedConcepts, setSelectedConcepts] = useState([])
  const [conceptSummaries, setConceptSummaries] = useState({})
  const [conceptFilter, setConceptFilter] = useState('all')
  const [conceptSort, setConceptSort] = useState('path')
  const [showHiddenConcepts, setShowHiddenConcepts] = useState(false)

  const [strategy, setStrategy] = useState('learning_path') // 'learning_path' | 'weakest_first' | 'pick'

  const [sessionId, setSessionId] = useState(null)
  const [turns, setTurns] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const endRef = useRef(null)

  useEffect(() => {
    if (endRef.current) endRef.current.scrollIntoView({ behavior: 'smooth' })
  }, [turns])

  // Load notes metadata and default to the first ready note
  useEffect(() => {
    if (typeof window === 'undefined') return
    const stored = loadNotesMetadata()
    setNotes(stored)
    const ready = stored.filter((n) => !n.status || n.status === 'Ready' || mapJobStatusToLabel(n.status) === 'Ready')
    if (ready.length > 0) {
      setSelectedNoteIds([ready[0].id])
    }
  }, [])

  // Fetch concept list for selected notes
  useEffect(() => {
    if (!token) return
    if (selectedNoteIds.length === 0) {
      setConcepts([])
      setConceptsError(null)
      setConceptFeedbackError(null)
      setConceptsLoading(false)
      setSelectedConcepts([])
      setConceptSummaries({})
      return
    }

    const ids = (selectedNoteIds || []).filter(Boolean)
    if (ids.length === 0) {
      setConcepts([])
      setConceptsError(null)
      setConceptsLoading(false)
      setSelectedConcepts([])
      setConceptSummaries({})
      return
    }

    let cancelled = false
    async function fetchConcepts() {
      setConceptsLoading(true)
      setConceptsError(null)
      setConceptFeedbackError(null)
      try {
        const res = await fetch(`${API_BASE}/api/resources/concepts`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: token ? `Bearer ${token}` : 'Bearer test-token',
          },
          body: JSON.stringify({ resource_ids: ids }),
        })
        if (!res.ok) {
          throw new Error(`Concepts HTTP ${res.status}`)
        }
        const data = await res.json()
        if (!cancelled) {
          setConcepts(Array.isArray(data) ? data : [])
          setConceptSummaries({})
          setSelectedConcepts([])
        }
      } catch (e) {
        if (!cancelled) {
          setConceptsError(String(e))
          setConcepts([])
        }
      } finally {
        if (!cancelled) {
          setConceptsLoading(false)
        }
      }
    }

    void fetchConcepts()
    return () => {
      cancelled = true
    }
  }, [token, JSON.stringify(selectedNoteIds)])

  const readyNotes = useMemo(
    () => notes.filter((n) => !n.status || n.status === 'Ready' || mapJobStatusToLabel(n.status) === 'Ready'),
    [notes],
  )

  const visibleConcepts = useMemo(() => {
    if (!Array.isArray(concepts) || concepts.length === 0) return []
    let items = concepts.slice()

    if (!showHiddenConcepts) {
      items = items.filter((c) => !c.hidden)
    }

    if (conceptFilter === 'weak') {
      items = items.filter((c) => !c.level || c.level === 'beginner' || c.level === 'developing')
    }

    const compareAlpha = (a, b) => {
      const an = (a.concept || '').toLowerCase()
      const bn = (b.concept || '').toLowerCase()
      if (an < bn) return -1
      if (an > bn) return 1
      return 0
    }

    if (conceptSort === 'path') {
      items.sort((a, b) => {
        const ap = typeof a.path_index === 'number' ? a.path_index : Number.MAX_SAFE_INTEGER
        const bp = typeof b.path_index === 'number' ? b.path_index : Number.MAX_SAFE_INTEGER
        if (ap !== bp) return ap - bp
        const am = typeof a.mastery === 'number' ? a.mastery : 1
        const bm = typeof b.mastery === 'number' ? b.mastery : 1
        if (am !== bm) return am - bm
        return compareAlpha(a, b)
      })
      return items
    }

    if (conceptSort === 'weakest_first') {
      items.sort((a, b) => {
        const am = typeof a.mastery === 'number' ? a.mastery : 1
        const bm = typeof b.mastery === 'number' ? b.mastery : 1
        if (am !== bm) return am - bm
        const ap = typeof a.path_index === 'number' ? a.path_index : Number.MAX_SAFE_INTEGER
        const bp = typeof b.path_index === 'number' ? b.path_index : Number.MAX_SAFE_INTEGER
        if (ap !== bp) return ap - bp
        return compareAlpha(a, b)
      })
      return items
    }

    // default alpha
    items.sort(compareAlpha)
    return items
  }, [concepts, conceptFilter, conceptSort, showHiddenConcepts])

  function masteryPillStyle(level) {
    if (level === 'beginner') {
      return { background: '#fee2e2', color: '#991b1b', borderRadius: 9999, padding: '2px 8px', fontSize: 11 }
    }
    if (level === 'developing') {
      return { background: '#fef3c7', color: '#92400e', borderRadius: 9999, padding: '2px 8px', fontSize: 11 }
    }
    if (level === 'proficient') {
      return { background: '#dcfce7', color: '#166534', borderRadius: 9999, padding: '2px 8px', fontSize: 11 }
    }
    if (level === 'mastering') {
      return { background: '#e0f2fe', color: '#075985', borderRadius: 9999, padding: '2px 8px', fontSize: 11 }
    }
    return { background: '#e5e7eb', color: '#374151', borderRadius: 9999, padding: '2px 8px', fontSize: 11 }
  }

  function handleToggleConceptSelection(concept) {
    if (!concept) return
    const name = String(concept).trim()
    if (!name) return
    setSelectedConcepts((prev) => {
      if (prev.includes(name)) {
        return prev.filter((c) => c !== name)
      }
      return [...prev, name]
    })
  }

  async function handleToggleConceptSummary(conceptObj) {
    if (!conceptObj) return
    const key = String(conceptObj.canonical || conceptObj.concept || '').trim()
    if (!key) return

    let shouldFetch = false
    const ids = (selectedNoteIds || []).filter(Boolean)
    if (ids.length === 0) {
      return
    }

    setConceptSummaries((prev) => {
      const existing = prev[key] || {}
      const nextExpanded = !existing.expanded
      const next = {
        ...prev,
        [key]: {
          ...existing,
          expanded: nextExpanded,
        },
      }
      if (nextExpanded && !existing.summary && !existing.loading) {
        shouldFetch = true
        next[key] = {
          ...next[key],
          loading: true,
          error: null,
        }
      }
      return next
    })

    if (!shouldFetch) return

    try {
      const res = await fetch(`${API_BASE}/api/resources/concepts/summary`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: token ? `Bearer ${token}` : 'Bearer test-token',
        },
        body: JSON.stringify({
          concept: conceptObj.canonical || conceptObj.concept,
          resource_ids: ids,
          max_chunks: 6,
        }),
      })
      if (!res.ok) {
        throw new Error(`Concept summary HTTP ${res.status}`)
      }
      const data = await res.json()
      const summaryText = typeof data.summary === 'string' ? data.summary : ''
      setConceptSummaries((prev) => ({
        ...prev,
        [key]: {
          ...(prev[key] || {}),
          loading: false,
          error: null,
          summary: summaryText,
          chunk_ids: Array.isArray(data.chunk_ids) ? data.chunk_ids : [],
          expanded: true,
        },
      }))
    } catch (e) {
      setConceptSummaries((prev) => ({
        ...prev,
        [key]: {
          ...(prev[key] || {}),
          loading: false,
          error: String(e),
          summary: (prev[key] && prev[key].summary) || '',
        },
      }))
    }
  }

  function computeTargetsForStrategy() {
    const items = visibleConcepts
    if (!items || items.length === 0) return []

    if (strategy === 'pick') {
      return (selectedConcepts || []).map((c) => String(c).trim()).filter(Boolean)
    }

    if (strategy === 'weakest_first') {
      const sorted = items.slice().sort((a, b) => {
        const am = typeof a.mastery === 'number' ? a.mastery : 1
        const bm = typeof b.mastery === 'number' ? b.mastery : 1
        if (am !== bm) return am - bm
        const ap = typeof a.path_index === 'number' ? a.path_index : Number.MAX_SAFE_INTEGER
        const bp = typeof b.path_index === 'number' ? b.path_index : Number.MAX_SAFE_INTEGER
        return ap - bp
      })
      return sorted.slice(0, 5).map((c) => c.concept).filter(Boolean)
    }

    // learning_path
    const sorted = items.slice().sort((a, b) => {
      const ap = typeof a.path_index === 'number' ? a.path_index : Number.MAX_SAFE_INTEGER
      const bp = typeof b.path_index === 'number' ? b.path_index : Number.MAX_SAFE_INTEGER
      if (ap !== bp) return ap - bp
      const am = typeof a.mastery === 'number' ? a.mastery : 1
      const bm = typeof b.mastery === 'number' ? b.mastery : 1
      return am - bm
    })
    return sorted.slice(0, 5).map((c) => c.concept).filter(Boolean)
  }

  async function ensureSession(targetsOverride) {
    if (sessionId) return sessionId

    if (!selectedNoteIds || selectedNoteIds.length === 0) {
      setError('Select at least one ready note before starting.')
      throw new Error('no_notes_selected')
    }

    const resourceIds = (selectedNoteIds || []).filter(Boolean)
    const targets = (targetsOverride || computeTargetsForStrategy() || [])
      .map((c) => String(c).trim())
      .filter(Boolean)

    if (!targets || targets.length === 0) {
      setError('Select at least one concept or choose a strategy that yields concepts.')
      throw new Error('no_targets_selected')
    }

    const body = {
      mode: 'personal_notes',
      agent_action_mode: 'step_by_step',
      resource_ids: resourceIds,
      target_concepts: targets,
    }

    const res = await fetch(`${API_BASE}/api/tutor/session/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: token ? `Bearer ${token}` : 'Bearer test-token',
      },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      throw new Error(`Session start HTTP ${res.status}`)
    }
    const data = await res.json()
    const sid = data.session_id
    setSessionId(sid)
    return sid
  }

  async function sendToTutor(
    text,
    {
      stepControl = null,
      mcqAnswer = null,
      confirmedAction = null,
      actionOverride = null,
      suppressUserTurn = false,
    } = {},
  ) {
    const trimmed = (text || '').trim()
    const isControlOnly = !!stepControl || !!mcqAnswer || !!confirmedAction

    if (!trimmed && !isControlOnly) return
    if (loading) return

    if (!selectedNoteIds || selectedNoteIds.length === 0) {
      setError('Select at least one ready note before starting.')
      return
    }

    setError(null)
    if (!isControlOnly && !suppressUserTurn) {
      setInput('')
      const userContent = trimmed
      if (userContent) {
        setTurns((prev) => [
          ...prev,
          {
            id: newId(),
            role: 'user',
            content: userContent,
          },
        ])
      }
    }

    setLoading(true)
    try {
      const sid = await ensureSession()
      const payload = {
        message: isControlOnly ? '' : trimmed,
      }
      if (confirmedAction) {
        payload.confirmed_action = confirmedAction
      }
      if (stepControl) {
        payload.step_control = stepControl
      }
      if (mcqAnswer) {
        payload.mcq_answer = mcqAnswer
      }
      if (actionOverride && actionOverride.type && actionOverride.type !== 'auto') {
        payload.action_override = actionOverride
      }

      const res = await fetch(`${API_BASE}/api/tutor/pedagogical/session/${sid}/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: token ? `Bearer ${token}` : 'Bearer test-token',
        },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        throw new Error(`Pedagogical tutor HTTP ${res.status}`)
      }
      const data = await res.json()

      const messages = Array.isArray(data.messages) ? data.messages : []
      const tutorText = messages.map((m) => m.content || '').join('\n').trim() || '(no response)'
      const meta = {
        uiMode: data.ui_mode || 'free_text',
        mcq: data.mcq_payload || null,
        debug: data.debug || {},
      }

      setTurns((prev) => [
        ...prev,
        {
          id: newId(),
          role: 'tutor',
          content: tutorText,
          meta,
        },
      ])
    } catch (e) {
      setError(String(e))
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

  async function handleSend() {
    const trimmed = (input || '').trim()
    if (!trimmed) return
    await sendToTutor(trimmed)
  }

  async function handleContinue() {
    await sendToTutor('', { confirmedAction: 'continue' })
  }

  async function handleSkipToQuiz() {
    await sendToTutor('', { stepControl: { type: 'skip_to_quiz' } })
  }

  async function handleNextConcept() {
    await sendToTutor('', { stepControl: { type: 'skip_to_next_concept' } })
  }

  async function handleReplanConcept() {
    await sendToTutor('', { stepControl: { type: 'replan_concept' } })
  }

  async function handleEndSession() {
    await sendToTutor('', { confirmedAction: 'end', actionOverride: { type: 'session_end' } })
  }

  async function handleMcqAnswer(turnMeta, opt) {
    if (!turnMeta || !opt) return
    await sendToTutor('', {
      mcqAnswer: {
        question_id: turnMeta.mcq.question_id || 'q1',
        option_id: opt.id,
      },
    })
  }

  const latestTurn = turns.length > 0 ? turns[turns.length - 1] : null
  const latestMcq = latestTurn && latestTurn.meta && latestTurn.meta.uiMode === 'mcq' ? latestTurn.meta.mcq : null
  const latestDebug = latestTurn && latestTurn.meta ? latestTurn.meta.debug : null
  const latestConceptPlan = latestDebug && latestDebug.concept_plan ? latestDebug.concept_plan : null
  const currentConceptId =
    (latestDebug && latestDebug.current_concept_id) || (latestConceptPlan && latestConceptPlan.concept_id) || null

  return (
    <main style={{ padding: 24, maxWidth: 960, margin: '0 auto' }}>
      <h1 style={{ fontSize: 26, marginBottom: 8 }}>Pedagogical Tutor MDP</h1>
      <p style={{ marginTop: 0, marginBottom: 16, color: '#4b5563', maxWidth: 720 }}>
        This view uses the new three-layer MDP stack (session, concept, pedagogical tutor). You can send free-text
        questions, use the control buttons to steer the flow, and answer MCQs when they appear.
      </p>

      <section
        style={{
          marginBottom: 16,
          padding: '10px 12px',
          borderRadius: 8,
          background: '#eff6ff',
          border: '1px solid #bfdbfe',
          fontSize: 12,
          color: '#1d4ed8',
        }}
      >
        <div style={{ marginBottom: 6 }}>
          <strong>Session:</strong> {sessionId || '(not started yet)'}
        </div>
        <div style={{ marginBottom: 6 }}>
          <strong>Notes: </strong>
          <select
            value={selectedNoteIds[0] || ''}
            onChange={(e) => setSelectedNoteIds(e.target.value ? [e.target.value] : [])}
            style={{ marginLeft: 8, padding: '4px 8px', fontSize: 12, minWidth: 220 }}
          >
            <option value="">Select a ready note…</option>
            {readyNotes.map((n) => (
              <option key={n.id} value={n.id}>
                {n.title || n.filename || n.id}
              </option>
            ))}
          </select>
        </div>
        <div>
          <strong>Concept strategy: </strong>
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
            style={{ marginLeft: 8, padding: '4px 8px', fontSize: 12 }}
          >
            <option value="learning_path">Learning path</option>
            <option value="weakest_first">Weakest first</option>
            <option value="pick">Pick manually</option>
          </select>
        </div>
      </section>

      <section style={{ marginBottom: 16, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        <button
          type="button"
          onClick={handleContinue}
          disabled={loading}
          style={{ padding: '6px 10px', fontSize: 13, borderRadius: 6, border: '1px solid #22c55e', background: '#dcfce7' }}
        >
          Next step
        </button>
        <button
          type="button"
          onClick={handleSkipToQuiz}
          disabled={loading}
          style={{ padding: '6px 10px', fontSize: 13, borderRadius: 6, border: '1px solid #f97316', background: '#ffedd5' }}
        >
          Skip to quiz
        </button>
        <button
          type="button"
          onClick={handleNextConcept}
          disabled={loading}
          style={{ padding: '6px 10px', fontSize: 13, borderRadius: 6, border: '1px solid #3b82f6', background: '#dbeafe' }}
        >
          Next concept
        </button>
        <button
          type="button"
          onClick={handleReplanConcept}
          disabled={loading}
          style={{ padding: '6px 10px', fontSize: 13, borderRadius: 6, border: '1px solid #a855f7', background: '#f3e8ff' }}
        >
          Replan concept
        </button>
        <button
          type="button"
          onClick={handleEndSession}
          disabled={loading}
          style={{ padding: '6px 10px', fontSize: 13, borderRadius: 6, border: '1px solid #ef4444', background: '#fee2e2' }}
        >
          End session
        </button>
      </section>

      {error && (
        <div style={{ marginBottom: 12, color: '#b91c1c', fontSize: 13 }}>Error: {String(error)}</div>
      )}

      {latestConceptPlan && (
        <section
          style={{
            marginBottom: 16,
            padding: 10,
            borderRadius: 8,
            border: '1px solid #e5e7eb',
            background: '#f9fafb',
            fontSize: 12,
            color: '#374151',
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 4 }}>SRL concept plan</div>
          <div style={{ marginBottom: 4 }}>
            <strong>Concept:</strong> {currentConceptId || '(unknown)'}
            {typeof latestConceptPlan.plan_index === 'number' &&
              typeof latestConceptPlan.plan_length === 'number' &&
              latestConceptPlan.plan_length > 0 && (
                <span style={{ marginLeft: 8 }}>
                  Step {Math.min(latestConceptPlan.plan_index + 1, latestConceptPlan.plan_length)} of{' '}
                  {latestConceptPlan.plan_length}
                </span>
              )}
          </div>
          {Array.isArray(latestConceptPlan.steps) && latestConceptPlan.steps.length > 0 && (
            <ol style={{ margin: 0, paddingLeft: 16 }}>
              {latestConceptPlan.steps.map((step, idx) => {
                const isCurrent =
                  typeof latestConceptPlan.plan_index === 'number' && idx === latestConceptPlan.plan_index
                return (
                  <li key={idx} style={{ marginBottom: 2, fontWeight: isCurrent ? 600 : 400 }}>
                    <span style={{ marginRight: 4 }}>{step.step_type || 'STEP'}</span>
                    {step.subgoal && <span>- {step.subgoal}</span>}
                  </li>
                )
              })}
            </ol>
          )}
        </section>
      )}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
        <section
          style={{
            flex: '1 1 0%',
            minWidth: 0,
            marginBottom: 16,
            padding: 12,
            borderRadius: 8,
            border: '1px solid #e5e7eb',
            maxHeight: 420,
            overflowY: 'auto',
            background: '#ffffff',
          }}
        >
          {turns.map((t) => (
            <div key={t.id} style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#6b7280' }}>{t.role === 'user' ? 'You' : 'Tutor'}</div>
              <div style={{ whiteSpace: 'pre-wrap', fontSize: 14 }}>{t.content}</div>
            </div>
          ))}
          <div ref={endRef} />
        </section>

        {!sessionId && (
          <ConceptsSidebar
            readyNotes={readyNotes}
            selectedNoteIds={selectedNoteIds}
            concepts={concepts}
            conceptsLoading={conceptsLoading}
            conceptsError={conceptsError}
            conceptFeedbackError={conceptFeedbackError}
            visibleConcepts={visibleConcepts}
            conceptFilter={conceptFilter}
            onChangeConceptFilter={setConceptFilter}
            conceptSort={conceptSort}
            onChangeConceptSort={setConceptSort}
            showHiddenConcepts={showHiddenConcepts}
            onChangeShowHidden={setShowHiddenConcepts}
            selectedConcepts={selectedConcepts}
            loading={loading}
            latestTutorConcept={null}
            masteryPillStyle={masteryPillStyle}
            conceptSummaries={conceptSummaries}
            onToggleConceptSummary={handleToggleConceptSummary}
            onToggleConceptSelection={handleToggleConceptSelection}
            onTeachConcept={() => {}}
            onReviewSelectedConcepts={() => {}}
            onHideConcept={() => {}}
          />
        )}
      </div>

      {latestMcq && (
        <section
          style={{
            marginBottom: 16,
            padding: 12,
            borderRadius: 8,
            border: '1px solid #fed7aa',
            background: '#fffbeb',
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Quick check</div>
          <div style={{ marginBottom: 8 }}>{latestMcq.question}</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {(latestMcq.options || []).map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => handleMcqAnswer({ mcq: latestMcq }, opt)}
                disabled={loading}
                style={{
                  textAlign: 'left',
                  padding: '6px 8px',
                  borderRadius: 6,
                  border: '1px solid #e5e7eb',
                  background: '#ffffff',
                  fontSize: 13,
                }}
              >
                <strong style={{ marginRight: 6 }}>{opt.id}.</strong>
                {opt.text}
              </button>
            ))}
          </div>
        </section>
      )}

      {latestDebug && (
        <section
          style={{
            marginBottom: 16,
            padding: 10,
            borderRadius: 8,
            border: '1px dashed #d1d5db',
            background: '#f9fafb',
            fontSize: 11,
            color: '#4b5563',
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Debug (MDP actions)</div>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{JSON.stringify(latestDebug, null, 2)}</pre>
        </section>
      )}

      <section style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              void handleSend()
            }
          }}
          placeholder="Ask a question or say what you want to work on..."
          style={{
            flex: 1,
            padding: '8px 10px',
            borderRadius: 6,
            border: '1px solid #d1d5db',
            fontSize: 14,
          }}
        />
        <button
          type="button"
          onClick={handleSend}
          disabled={loading}
          style={{
            padding: '8px 14px',
            borderRadius: 6,
            border: '1px solid #2563eb',
            background: '#2563eb',
            color: '#ffffff',
            fontSize: 14,
            fontWeight: 500,
          }}
        >
          Send
        </button>
      </section>
    </main>
  )
}
