"use client"

import { useEffect, useMemo, useRef, useState } from 'react'
import { useAuth } from '../../hooks/useAuth'
import { API_BASE } from '../../lib/api'
import { loadNotesMetadata, mapJobStatusToLabel } from '../../lib/notes'
import ConceptsSidebar from '../ConceptsSidebar'
import PlanControls from './PlanControls'
import ConversationPanel from './ConversationPanel'

const STATIC_TUTOR_MODELS = [
  { id: 'openai/gpt-5-mini-2025-08-07', display_name: 'openai/gpt-5-mini-2025-08-07' },
  { id: 'anthropic/claude-haiku-4.5', display_name: 'anthropic/claude-haiku-4.5' },
  { id: 'anthropic/claude-sonnet-4.5', display_name: 'anthropic/claude-sonnet-4.5' },
  { id: 'deepseek/deepseek-non-thinking-v3.2-exp', display_name: 'deepseek/deepseek-non-thinking-v3.2-exp' },
  { id: 'google/gemini-2.5-pro', display_name: 'google/gemini-2.5-pro' },
  { id: 'moonshot/kimi-k2-turbo-preview', display_name: 'moonshot/kimi-k2-turbo-preview' },
  { id: 'x-ai/grok-4-fast-reasoning', display_name: 'x-ai/grok-4-fast-reasoning' },
]

function newId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export default function StepTutorPage() {
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
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [input, setInput] = useState('')
  const [planApproved, setPlanApproved] = useState(false)
  const [highLevelPlan, setHighLevelPlan] = useState('')

  const [models, setModels] = useState([])
  const [selectedModelId, setSelectedModelId] = useState('')
  const [nextActionOverride, setNextActionOverride] = useState('auto')

  const endRef = useRef(null)

  useEffect(() => {
    if (typeof window === 'undefined') return
    const stored = loadNotesMetadata()
    setNotes(stored)
    const ready = stored.filter((n) => !n.status || n.status === 'Ready' || mapJobStatusToLabel(n.status) === 'Ready')
    if (ready.length > 0) {
      setSelectedNoteIds([ready[0].id])
    }
  }, [])

  useEffect(() => {
    async function fetchModels() {
      try {
        const res = await fetch(`${API_BASE}/api/tutor/models`, {
          headers: {
            Authorization: token ? `Bearer ${token}` : 'Bearer test-token',
          },
        })
        if (!res.ok) {
          setModels(STATIC_TUTOR_MODELS)
          return
        }
        const data = await res.json()
        const list = Array.isArray(data) && data.length > 0 ? data : STATIC_TUTOR_MODELS
        setModels(list)
        if (!selectedModelId && list && list.length > 0) {
          setSelectedModelId(list[0].id)
        }
      } catch (e) {
        setModels(STATIC_TUTOR_MODELS)
      }
    }
    if (token) {
      void fetchModels()
    }
  }, [token, selectedModelId])

  useEffect(() => {
    if (endRef.current) {
      endRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [turns])

  const latestPlan = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      const t = turns[i]
      if (t.role === 'tutor' && t.meta && t.meta.plan) {
        return t.meta.plan
      }
    }
    return null
  }, [turns])

  const latestStep = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      const t = turns[i]
      if (t.role === 'tutor' && t.meta && t.meta.step) {
        return t.meta.step
      }
    }
    return null
  }, [turns])

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

  const latestTutorConcept = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      const t = turns[i]
      if (t.role === 'tutor' && t.meta && t.meta.concept) {
        return String(t.meta.concept)
      }
    }
    return null
  }, [turns])

  const latestSrlPlan = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      const t = turns[i]
      if (t.role === 'tutor' && t.meta && t.meta.srlPlan) {
        return t.meta.srlPlan
      }
    }
    return null
  }, [turns])

  const latestSrlNextStep = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      const t = turns[i]
      if (t.role === 'tutor' && t.meta && t.meta.srlNextStep) {
        return t.meta.srlNextStep
      }
    }
    return null
  }, [turns])

  const currentConcept = useMemo(() => {
    if (latestTutorConcept) return latestTutorConcept
    const plan = latestPlan || latestSrlPlan
    if (plan && Array.isArray(plan.target_sequence) && plan.target_sequence.length > 0) {
      return plan.target_sequence[0]
    }
    return null
  }, [latestTutorConcept, latestPlan, latestSrlPlan])

  const hasPlan = !!latestSrlPlan
  const hasNextStep = !!latestSrlNextStep

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

  async function ensureSession(targetConcepts) {
    if (sessionId) return sessionId

    const apiMode = 'personal_notes'
    const apiActionMode = 'step_by_step'
    const resourceIds = (selectedNoteIds || []).filter(Boolean)

    const body = {
      mode: apiMode,
      agent_action_mode: apiActionMode,
      resource_ids: resourceIds,
    }

    const targets = (targetConcepts || []).map((c) => String(c).trim()).filter(Boolean)
    if (targets.length > 0) {
      body.target_concepts = targets
    }

    if (models && models.length > 0) {
      const fallback = models[0].id
      const mid = selectedModelId || fallback
      if (mid) {
        body.model_strategy = { type: 'single', model_id: mid }
      }
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
      throw new Error(`Tutor session start HTTP ${res.status}`)
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
      targetConcepts = null,
      suppressUserTurn = false,
      planType = null, // 'high_level' for the initial cross-concept study plan
    } = {},
  ) {
    const trimmed = (text || '').trim()
    const isControlOnly = !!stepControl || !!mcqAnswer || !!confirmedAction

    if (!trimmed && !isControlOnly) return
    if (loading) return

    if (!selectedNoteIds || selectedNoteIds.length === 0) {
      setError('Select at least one ready note before starting step-by-step mode.')
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
      const sid = await ensureSession(targetConcepts)
      const payload = {
        message: isControlOnly ? '' : trimmed,
        agent_action_mode: 'step_by_step',
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

      const res = await fetch(`${API_BASE}/api/tutor/session/${sid}/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: token ? `Bearer ${token}` : 'Bearer test-token',
        },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        throw new Error(`Tutor agent HTTP ${res.status}`)
      }
      const data = await res.json()

      const tutorContent = data.response || data.answer || ''

      if (planType === 'high_level') {
        setHighLevelPlan(tutorContent)
      }

      const tutorTurn = {
        id: newId(),
        role: 'tutor',
        content: tutorContent,
        meta: {
          actionType: data.action_type,
          confidence: data.confidence,
          concept: data.concept,
          level: data.level,
          intent: data.intent,
          affect: data.affect,
          sourceChunks: data.source_chunk_ids || [],
          srlPlan: data.srl_plan,
          srlNextStep: data.srl_next_step,
          // New canonical step/plan projections from StepEngine
          step: data.step || null,
          plan: data.plan || null,
          quizPhase: data.quiz_phase,
          quizQuestionIndex: data.quiz_question_index,
          quizMaxQuestions: data.quiz_max_questions,
          mcq: data.mcq,
        },
      }
      setTurns((prev) => [...prev, tutorTurn])
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

  async function handleGeneratePlan() {
    if (!selectedNoteIds || selectedNoteIds.length === 0) {
      setError('Select at least one ready note before generating a plan.')
      return
    }
    const targets = computeTargetsForStrategy()
    if (!targets || targets.length === 0) {
      setError('Select at least one concept or choose a strategy that yields concepts.')
      return
    }

    setError(null)
    setPlanApproved(false)
    setHighLevelPlan('')
    setTurns([])
    setSessionId(null)

    const preview = targets.slice(0, 5).join(', ')
    const msg =
      targets.length === 1
        ? `Please create a step-by-step study plan to learn ${targets[0]} from my notes.`
        : `Please create a step-by-step study plan to learn these concepts from my notes: ${preview}.`

    await sendToTutor(msg, { targetConcepts: targets, suppressUserTurn: true, planType: 'high_level' })
  }

  async function handleContinue(actionOverrideType) {
    const override = actionOverrideType && actionOverrideType !== 'auto' ? { type: actionOverrideType } : null
    await sendToTutor('', { confirmedAction: 'continue', actionOverride: override })
  }

  async function handleMcqAnswer(turnMeta, opt) {
    if (!turnMeta || !opt) return
    await sendToTutor('', {
      mcqAnswer: {
        question_id: turnMeta.mcq.question_id,
        option_id: opt.id,
      },
    })
  }

  async function handleSkipToQuiz() {
    await sendToTutor('', { stepControl: { type: 'skip_to_quiz' } })
  }

  async function handleNextConcept() {
    await sendToTutor('', { stepControl: { type: 'skip_to_next_concept' } })
  }

  async function handleEndSession() {
    await sendToTutor('', { confirmedAction: 'end', actionOverride: { type: 'session_end' } })
  }

  async function handleFreeTextSend() {
    const trimmed = (input || '').trim()
    if (!trimmed) return
    await sendToTutor(trimmed)
  }

  async function handleApprovePlan() {
    if (planApproved) return
    setPlanApproved(true)
    // Kick off SRL planning and execution for the first concept by
    // sending a continue control turn. The resulting response should
    // populate srl_plan and srl_next_step so the SRL plan card and
    // next-step card can render, after which the primary button
    // transitions to "Continue".
    await handleContinue(nextActionOverride)
  }

  return (
    <main>
      <div
        style={{
          marginBottom: 12,
          padding: '8px 12px',
          borderRadius: 8,
          background: '#eff6ff',
          border: '1px solid #bfdbfe',
          fontSize: 12,
          color: '#1d4ed8',
        }}
      >
        Step-by-step tutor grounded in your notes. First choose what to study, approve the plan, then use Continue to
        walk through the SRL steps.
      </div>

      <h1 style={{ fontSize: 26, marginBottom: 8 }}>Step-by-step Tutor</h1>
      <p style={{ marginTop: 0, marginBottom: 16, color: '#4b5563', maxWidth: 720 }}>
        This view is optimized for self-regulated learning with explicit study plans. Choose concepts, review the
        high-level plan, then execute it step-by-step.
      </p>

      {currentConcept && (
        <div
          style={{
            marginBottom: 12,
            padding: '6px 10px',
            borderRadius: 9999,
            background: '#ecfdf3',
            border: '1px solid #22c55e',
            fontSize: 12,
            color: '#166534',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <span style={{ fontWeight: 600 }}>Current concept:</span>
          <span>{currentConcept}</span>
        </div>
      )}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
        <PlanControls
          readyNotes={readyNotes}
          selectedNoteIds={selectedNoteIds}
          onChangeSelectedNoteIds={setSelectedNoteIds}
          models={models && models.length > 0 ? models : STATIC_TUTOR_MODELS}
          selectedModelId={selectedModelId}
          onChangeSelectedModelId={setSelectedModelId}
          strategy={strategy}
          onChangeStrategy={setStrategy}
          highLevelPlan={highLevelPlan}
          latestSrlPlan={latestSrlPlan}
          latestSrlNextStep={latestSrlNextStep}
          latestPlan={latestPlan}
          latestStep={latestStep}
          planApproved={planApproved}
          loading={loading}
          error={error}
          nextActionOverride={nextActionOverride}
          onChangeNextActionOverride={setNextActionOverride}
          onGeneratePlan={handleGeneratePlan}
          onApprovePlan={handleApprovePlan}
          onContinue={() => handleContinue(nextActionOverride)}
          onSkipToQuiz={handleSkipToQuiz}
          onNextConcept={handleNextConcept}
          onEndSession={handleEndSession}
        />

        <ConversationPanel
          turns={turns}
          loading={loading}
          input={input}
          onInputChange={setInput}
          onSendMessage={handleFreeTextSend}
          onMcqAnswer={handleMcqAnswer}
          endRef={endRef}
        />

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
          latestTutorConcept={latestTutorConcept}
          masteryPillStyle={masteryPillStyle}
          conceptSummaries={conceptSummaries}
          onToggleConceptSummary={handleToggleConceptSummary}
          onToggleConceptSelection={handleToggleConceptSelection}
          // Reuse sidebar actions as pure selection tools in this view
          onTeachConcept={() => {}}
          onReviewSelectedConcepts={() => {}}
          onHideConcept={() => {}}
        />
      </div>
    </main>
  )
}
