"use client"

import { useEffect, useMemo, useRef, useState } from 'react'
import { useAuth } from '../hooks/useAuth'
import { API_BASE } from '../lib/api'
import { loadNotesMetadata, mapJobStatusToLabel } from '../lib/notes'
import { Card, PrimaryButton, SecondaryButton } from '../components/ui'
import ConceptsSidebar from './ConceptsSidebar'

function newId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

const STATIC_TUTOR_MODELS = [
  { id: 'openai/gpt-5-mini-2025-08-07', display_name: 'openai/gpt-5-mini-2025-08-07' },
  { id: 'anthropic/claude-haiku-4.5', display_name: 'anthropic/claude-haiku-4.5' },
  { id: 'anthropic/claude-sonnet-4.5', display_name: 'anthropic/claude-sonnet-4.5' },
  { id: 'deepseek/deepseek-non-thinking-v3.2-exp', display_name: 'deepseek/deepseek-non-thinking-v3.2-exp' },
  { id: 'google/gemini-2.5-pro', display_name: 'google/gemini-2.5-pro' },
  { id: 'moonshot/kimi-k2-turbo-preview', display_name: 'moonshot/kimi-k2-turbo-preview' },
  { id: 'x-ai/grok-4-fast-reasoning', display_name: 'x-ai/grok-4-fast-reasoning' },
]

export default function TutorPage() {
  const { user, token } = useAuth({ requireAuth: true })
  const [mode, setMode] = useState('notes')
  const [actionMode, setActionMode] = useState('auto')
  const [tutorAction, setTutorAction] = useState('auto')
  const [input, setInput] = useState('What should I revise today?')
  const [pending, setPending] = useState(null)
  const [turns, setTurns] = useState([])
  const [notes, setNotes] = useState([])
  const [selectedNoteIds, setSelectedNoteIds] = useState([])
  const [concepts, setConcepts] = useState([])
  const [conceptsLoading, setConceptsLoading] = useState(false)
  const [conceptsError, setConceptsError] = useState(null)
  const [conceptFeedbackError, setConceptFeedbackError] = useState(null)
  const [selectedConcepts, setSelectedConcepts] = useState([])
  const [conceptSummaries, setConceptSummaries] = useState({})
  const [conceptFilter, setConceptFilter] = useState('all') // 'all' | 'weak'
  const [conceptSort, setConceptSort] = useState('weak_first') // 'weak_first' | 'path' | 'alpha'
  const [showHiddenConcepts, setShowHiddenConcepts] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [models, setModels] = useState([])
  const [modelStrategy, setModelStrategy] = useState('single') // 'single' | 'multi_rank'
  const [selectedModelId, setSelectedModelId] = useState('')
  const [selectedModelIds, setSelectedModelIds] = useState([])
  const [multiCandidates, setMultiCandidates] = useState(null)
  const [selectedCandidateId, setSelectedCandidateId] = useState(null)
  const [candidateRating, setCandidateRating] = useState({ correctness: 1, helpfulness: 1, coverage: 1 })
  const [stepStartMode, setStepStartMode] = useState('auto') // 'auto' | 'pick'
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

  const latestTutorConcept = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      const t = turns[i]
      if (t.role === 'tutor' && t.meta && t.meta.concept) {
        return String(t.meta.concept)
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

  const latestSrlPlan = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      const t = turns[i]
      if (t.role === 'tutor' && t.meta && t.meta.srlPlan) {
        return t.meta.srlPlan
      }
    }
    return null
  }, [turns])

  const latestQuizMeta = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      const t = turns[i]
      if (t.role === 'tutor' && t.meta && t.meta.quizPhase) {
        return t.meta
      }
    }
    return null
  }, [turns])

  function resetSessionAndConversation(nextMode, nextActionMode) {
    setSessionId(null)
    setTurns([])
    setPending(null)
    setError(null)
    if (typeof nextMode === 'string') setMode(nextMode)
    if (typeof nextActionMode === 'string') setActionMode(nextActionMode)
  }

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
        // ignore model fetch errors in UI
        setModels(STATIC_TUTOR_MODELS)
      }
    }
    if (token) {
      void fetchModels()
    }
  }, [token, selectedModelId])

  function authHeader() {
    return token ? `Bearer ${token}` : 'Bearer test-token'
  }

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

    if (conceptSort === 'alpha') {
      items.sort(compareAlpha)
      return items
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

    // default: weak_first by ascending mastery, then path, then alpha
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
  }, [concepts, conceptFilter, conceptSort, showHiddenConcepts])

  useEffect(() => {
    if (mode !== 'notes') {
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
            Authorization: authHeader(),
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, JSON.stringify(selectedNoteIds), token])

  async function ensureSession(targetConceptsOverride = null) {
    if (sessionId) return sessionId

    const apiMode = mode === 'notes' ? 'personal_notes' : 'general'
    const apiActionMode = actionMode === 'step' ? 'step_by_step' : 'auto'
    const resourceIds = apiMode === 'personal_notes' ? (selectedNoteIds || []).filter(Boolean) : []

    const body = {
      mode: apiMode,
      agent_action_mode: apiActionMode,
      resource_ids: resourceIds,
    }

    let targets = null
    if (Array.isArray(targetConceptsOverride) && targetConceptsOverride.length > 0) {
      targets = targetConceptsOverride.filter((c) => typeof c === 'string' && c.trim())
    } else if (apiMode === 'personal_notes' && selectedConcepts && selectedConcepts.length > 0) {
      targets = selectedConcepts.filter((c) => typeof c === 'string' && c.trim())
    }
    if (targets && targets.length > 0) {
      body.target_concepts = targets
    }

    if (modelStrategy === 'multi_rank') {
      const ids = selectedModelIds.filter(Boolean).slice(0, 3)
      if (ids.length < 2) {
        setError('Select at least two models for multi-model comparison.')
        throw new Error('multi_rank_requires_two_models')
      }
      body.model_strategy = { type: 'multi_rank', model_ids: ids }
    } else if (models && models.length > 0) {
      const fallback = models[0].id
      const mid = selectedModelId || fallback
      body.model_strategy = { type: 'single', model_id: mid }
    }

    const res = await fetch(`${API_BASE}/api/tutor/session/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: authHeader(),
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

  function handleStartStepSession() {
    if (actionMode !== 'step') {
      return
    }

    if (mode !== 'notes') {
      const msg = (input || '').trim() || 'What should I revise today?'
      if (!msg) return
      void sendToTutor(msg)
      return
    }

    if (stepStartMode === 'pick') {
      const targets = (selectedConcepts || []).map((c) => String(c).trim()).filter(Boolean)
      if (targets.length === 0) {
        setError('Select at least one concept in the Concepts panel to start step-by-step with chosen concepts.')
        return
      }
      setError(null)
      const preview = targets.slice(0, 5).join(', ')
      const msg =
        targets.length === 1
          ? `Please create a step-by-step study plan to learn ${targets[0]} from my notes.`
          : `Please create a step-by-step study plan to learn these concepts from my notes: ${preview}.`
      setInput('')
      setPending(null)
      void sendToTutor(msg, {
        targetConcepts: targets,
        userDisplayText: msg,
      })
      return
    }

    // Default: tutor chooses concepts based on mastery / path
    const msg = 'What should I revise today?'
    setError(null)
    setInput('')
    setPending(null)
    setSelectedConcepts([])
    void sendToTutor(msg, {
      targetConcepts: [],
      userDisplayText: msg,
    })
  }

  async function sendToTutor(text, isContinue = false, overrideType = null) {
    // Support both legacy positional args and an options object as the second
    // argument, so we can pass stepControl / mcqAnswer without breaking
    // existing call sites.
    let options = {}
    if (isContinue && typeof isContinue === 'object') {
      options = isContinue || {}
      isContinue = options.isContinue || false
      if (typeof options.overrideType === 'string') {
        overrideType = options.overrideType
      }
    }
    const { stepControl = null, mcqAnswer = null, userDisplayText = null, targetConcepts = null } = options

    const trimmed = (text || '').trim()
    const isControlOnly = !!stepControl || !!mcqAnswer || (actionMode === 'step' && isContinue)
    if (!trimmed && !isControlOnly) return
    if (loading) return

    if (multiCandidates) {
      setError('Choose a candidate response or cancel comparison before sending a new message.')
      return
    }

    if (mode === 'notes' && (!selectedNoteIds || selectedNoteIds.length === 0)) {
      setError('Select at least one ready note before starting the tutor in notes mode.')
      return
    }

    setError(null)
    setInput('')
    setPending(null)
    const userContent = isControlOnly ? '' : (userDisplayText || trimmed || (stepControl && stepControl.type) || '')
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

    setLoading(true)
    try {
      const sid = await ensureSession(targetConcepts)
      const payload = {
        message: isControlOnly ? '' : trimmed,
        agent_action_mode: actionMode === 'step' ? 'step_by_step' : 'auto',
      }
      // Only send confirmed_action='continue' when user clicks Continue button
      if (actionMode === 'step' && isContinue && !stepControl) {
        payload.confirmed_action = 'continue'
      }
      if (stepControl) {
        payload.step_control = stepControl
      }
      if (mcqAnswer) {
        payload.mcq_answer = mcqAnswer
      }
      if (overrideType) {
        payload.action_override = { type: overrideType }
      } else if (tutorAction && tutorAction !== 'auto') {
        payload.action_override = { type: tutorAction }
      }

      const res = await fetch(`${API_BASE}/api/tutor/session/${sid}/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: authHeader(),
        },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        throw new Error(`Tutor agent HTTP ${res.status}`)
      }
      const data = await res.json()

      if (data.model_strategy_type === 'multi_rank' && Array.isArray(data.candidates)) {
        setMultiCandidates({
          sessionId: sid,
          message: trimmed,
          candidates: data.candidates || [],
        })
        return
      }

      const tutorTurn = {
        id: newId(),
        role: 'tutor',
        content: data.response || data.answer || '',
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
          // New canonical step/plan projections from StepEngine (step mode)
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

  function handleSubmit() {
    if (actionMode === 'auto') {
      void sendToTutor(input)
    } else if (!pending) {
      const trimmed = input.trim()
      if (!trimmed) return
      setPending(trimmed)
    }
  }

  function handleConfirmStep() {
    if (!pending) return
    void sendToTutor(pending)
  }

  const activeModeLabel = mode === 'notes' ? 'Using: Your notes' : 'Using: General tutor'
  const activeActionLabel = actionMode === 'auto' ? 'Actions: Auto' : 'Actions: Step-by-step'
  const activeTutorActionLabel =
    tutorAction === 'auto' ? 'Tutor action: Auto (agent decides)' : `Tutor action: ${tutorAction}`
  let activeModelLabel = 'Model: default'
  if (models && models.length > 0) {
    if (modelStrategy === 'single') {
      const id = selectedModelId || models[0].id
      const m = models.find((mm) => mm.id === id) || models[0]
      activeModelLabel = `Model: ${m.display_name || m.id}`
    } else {
      if (!selectedModelIds || selectedModelIds.length === 0) {
        activeModelLabel = 'Models: (select up to 3)'
      } else {
        const names = selectedModelIds
          .map((id) => {
            const m = models.find((mm) => mm.id === id)
            return m ? m.display_name || m.id : id
          })
          .join(', ')
        activeModelLabel = `Models: ${names}`
      }
    }
  }

  function handleCandidateRatingChange(field, value) {
    setCandidateRating((prev) => ({ ...prev, [field]: value }))
  }

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

  function handleTeachConcept(concept) {
    if (!concept) return
    resetSessionAndConversation('notes', actionMode)
    setSelectedConcepts([concept])
    setInput(`Can we revise ${concept} from my notes?`)
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

  function handleReviewSelectedConcepts() {
    const targets = (selectedConcepts || []).map((c) => String(c).trim()).filter(Boolean)
    if (targets.length === 0) return
    resetSessionAndConversation('notes', actionMode)
    setSelectedConcepts(targets)
    if (targets.length === 1) {
      setInput(`Can we start learning ${targets[0]} from my notes?`)
    } else {
      const preview = targets.slice(0, 5).join(', ')
      setInput(`Can we plan a short study session and learn these concepts from my notes: ${preview}?`)
    }
  }

  async function handleHideConcept(conceptObj) {
    if (!conceptObj) return
    const key = String(conceptObj.canonical || conceptObj.concept || '').trim()
    if (!key) return

    const ids = (selectedNoteIds || []).filter(Boolean)
    if (ids.length === 0) {
      return
    }

    const confirmed = window.confirm(
      'Hide this concept from concept lists for these notes? This will not delete your notes or KG data; it only hides the concept for you.',
    )
    if (!confirmed) return

    try {
      setConceptFeedbackError(null)
      const res = await fetch(`${API_BASE}/api/resources/concepts/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: authHeader(),
        },
        body: JSON.stringify({
          resource_ids: ids,
          concept: key,
          feedback_type: 'hide',
        }),
      })
      if (!res.ok) {
        throw new Error(`Concept feedback HTTP ${res.status}`)
      }

      setConcepts((prev) => {
        if (!Array.isArray(prev) || prev.length === 0) return prev
        return prev.map((c) => {
          const cKey = String(c.canonical || c.concept || '').trim()
          if (cKey && cKey.toLowerCase() === key.toLowerCase()) {
            return { ...c, hidden: true }
          }
          return c
        })
      })
    } catch (e) {
      setConceptFeedbackError(String(e))
    }
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
          Authorization: authHeader(),
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

  async function handleAdoptCandidate() {
    if (!multiCandidates || selectedCandidateId === null || loading) return
    setError(null)
    setLoading(true)
    try {
      const body = {
        message: multiCandidates.message,
        chosen_id: selectedCandidateId,
        candidates: multiCandidates.candidates,
        rating: candidateRating,
      }
      const res = await fetch(`${API_BASE}/api/tutor/session/${multiCandidates.sessionId}/choose-candidate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: authHeader(),
        },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        throw new Error(`Tutor choose-candidate HTTP ${res.status}`)
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
          intent: data.intent,
          affect: data.affect,
          sourceChunks: data.source_chunk_ids || [],
          modelId: data.model_id,
          modelName: data.model_name,
        },
      }
      setTurns((prev) => [...prev, tutorTurn])
      setMultiCandidates(null)
      setSelectedCandidateId(null)
      setCandidateRating({ correctness: 1, helpfulness: 1, coverage: 1 })
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
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
        StudyAgent is a research prototype. Your interactions may be logged and used to improve the system. See the{' '}
        <a href="/policy" style={{ textDecoration: 'underline' }}>
          short policy
        </a>
        .
      </div>

      <h1 style={{ fontSize: 26, marginBottom: 8 }}>Tutor</h1>
      <p style={{ marginTop: 0, marginBottom: 16, color: '#4b5563', maxWidth: 640 }}>
        Chat with the tutor to revise concepts, ask questions, and get guidance. You can ground the tutor in your notes
        or use a general mode.
      </p>

      {actionMode === 'step' && latestStep && (
        <div
          style={{
            marginBottom: 12,
            padding: '6px 10px',
            borderRadius: 9999,
            display: 'inline-flex',
            gap: 8,
            alignItems: 'center',
            background: '#f3f4ff',
            border: '1px solid #c7d2fe',
            fontSize: 11,
            color: '#4338ca',
          }}
        >
          <span style={{ fontWeight: 600 }}>Step mode</span>
          {latestStep.type && (
            <span style={{ padding: '2px 8px', borderRadius: 9999, background: '#eef2ff', color: '#4f46e5' }}>
              {String(latestStep.type)}
            </span>
          )}
          {latestStep.phase && (
            <span style={{ padding: '2px 8px', borderRadius: 9999, background: '#eef2ff', color: '#4b5563' }}>
              {String(latestStep.phase)}
            </span>
          )}
        </div>
      )}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
        <Card
          title="Configuration"
          style={{ flex: '1 1 260px', minWidth: 260 }}
        >
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#6b7280' }}>
              Mode
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
              <SecondaryButton
                onClick={() => resetSessionAndConversation('notes', actionMode)}
                style={mode === 'notes' ? { background: '#2563eb0d', borderColor: '#2563eb', color: '#1d4ed8' } : null}
              >
                Study my notes
              </SecondaryButton>
              <SecondaryButton
                onClick={() => resetSessionAndConversation('general', actionMode)}
                style={
                  mode === 'general' ? { background: '#2563eb0d', borderColor: '#2563eb', color: '#1d4ed8' } : null
                }
              >
                General tutor
              </SecondaryButton>
            </div>
          </div>

          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#6b7280' }}>
              Action mode
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
              <SecondaryButton
                onClick={() => resetSessionAndConversation(mode, 'auto')}
                style={
                  actionMode === 'auto' ? { background: '#2563eb0d', borderColor: '#2563eb', color: '#1d4ed8' } : null
                }
              >
                Auto
              </SecondaryButton>
              <SecondaryButton
                onClick={() => resetSessionAndConversation(mode, 'step')}
                style={
                  actionMode === 'step' ? { background: '#2563eb0d', borderColor: '#2563eb', color: '#1d4ed8' } : null
                }
              >
                Step-by-step
              </SecondaryButton>
            </div>
            {actionMode === 'step' && (
              <p style={{ marginTop: 6, fontSize: 12, color: '#6b7280' }}>
                In step-by-step mode, you propose the next action and confirm before the tutor executes it.
              </p>
            )}
            {actionMode === 'step' && mode === 'notes' && (
              <div style={{ marginTop: 6 }}>
                <div style={{ fontSize: 12, color: '#4b5563', marginBottom: 4 }}>Step-by-step start</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <SecondaryButton
                    onClick={() => setStepStartMode('auto')}
                    style={
                      stepStartMode === 'auto'
                        ? { background: '#2563eb0d', borderColor: '#2563eb', color: '#1d4ed8' }
                        : null
                    }
                  >
                    Tutor chooses concepts
                  </SecondaryButton>
                  <SecondaryButton
                    onClick={() => setStepStartMode('pick')}
                    style={
                      stepStartMode === 'pick'
                        ? { background: '#2563eb0d', borderColor: '#2563eb', color: '#1d4ed8' }
                        : null
                    }
                  >
                    I choose concepts
                  </SecondaryButton>
                </div>
              </div>
            )}
          </div>

          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#6b7280' }}>
              Tutor action
            </div>
            <select
              value={tutorAction}
              onChange={(e) => setTutorAction(e.target.value)}
              style={{
                marginTop: 6,
                width: '100%',
                padding: '7px 10px',
                borderRadius: 8,
                border: '1px solid #d1d5db',
                fontSize: 14,
              }}
            >
              <option value="auto">auto (agent decides)</option>
              <option value="explain">explain</option>
              <option value="ask">ask</option>
              <option value="hint">hint</option>
              <option value="reflect">reflect</option>
              <option value="worked_example">worked_example</option>
              <option value="review">review</option>
            </select>
          </div>

          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#6b7280' }}>
              Model strategy
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 6, flexWrap: 'wrap' }}>
              <SecondaryButton
                onClick={() => {
                  resetSessionAndConversation(mode, actionMode)
                  setModelStrategy('single')
                }}
                style={
                  modelStrategy === 'single'
                    ? { background: '#2563eb0d', borderColor: '#2563eb', color: '#1d4ed8' }
                    : null
                }
              >
                Single model
              </SecondaryButton>
              <SecondaryButton
                onClick={() => {
                  resetSessionAndConversation(mode, actionMode)
                  setModelStrategy('multi_rank')
                }}
                style={
                  modelStrategy === 'multi_rank'
                    ? { background: '#2563eb0d', borderColor: '#2563eb', color: '#1d4ed8' }
                    : null
                }
              >
                Multi-model (up to 3)
              </SecondaryButton>
            </div>
            {modelStrategy === 'single' && (
              <div style={{ marginTop: 6 }}>
                {models.length === 0 ? (
                  <p style={{ fontSize: 13, color: '#6b7280' }}>Using backend default tutor model.</p>
                ) : (
                  <select
                    value={selectedModelId}
                    onChange={(e) => {
                      setSelectedModelId(e.target.value)
                      resetSessionAndConversation(mode, actionMode)
                    }}
                    style={{
                      marginTop: 4,
                      width: '100%',
                      padding: '7px 10px',
                      borderRadius: 8,
                      border: '1px solid #d1d5db',
                      fontSize: 14,
                    }}
                  >
                    {models.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.display_name || m.id}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )}
            {modelStrategy === 'multi_rank' && (
              <div style={{ marginTop: 6 }}>
                {models.length === 0 ? (
                  <p style={{ fontSize: 13, color: '#6b7280' }}>No registered models; comparison mode falls back to default.</p>
                ) : (
                  <>
                    <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>
                      Select up to 3 models to compare.
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                      {models.map((m) => (
                        <label key={m.id} style={{ fontSize: 13, color: '#374151' }}>
                          <input
                            type="checkbox"
                            checked={selectedModelIds.includes(m.id)}
                            onChange={(e) => {
                              const checked = e.target.checked
                              setSelectedModelIds((prev) => {
                                if (checked) {
                                  const next = [...prev, m.id]
                                  return next.slice(0, 3)
                                }
                                return prev.filter((id) => id !== m.id)
                              })
                              resetSessionAndConversation(mode, actionMode)
                            }}
                            style={{ marginRight: 6 }}
                          />
                          {m.display_name || m.id}
                        </label>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          {mode === 'notes' && (
            <div style={{ marginTop: 4 }}>
              <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#6b7280' }}>
                Note selection
              </div>
              {readyNotes.length === 0 ? (
                <p style={{ marginTop: 6, fontSize: 13, color: '#6b7280' }}>
                  No ready notes found. Upload a PDF or slides on the <a href="/notes">Notes</a> page, then refresh this
                  page.
                </p>
              ) : (
                <div
                  style={{
                    marginTop: 6,
                    maxHeight: 180,
                    overflow: 'auto',
                    borderRadius: 8,
                    border: '1px solid #d1d5db',
                    padding: 8,
                  }}
                >
                  {readyNotes.map((n) => {
                    const checked = selectedNoteIds.includes(n.id)
                    const statusLabel = mapJobStatusToLabel(n.status)
                    return (
                      <label
                        key={n.id}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          gap: 8,
                          padding: '4px 4px',
                          fontSize: 13,
                          color: '#374151',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={(e) => {
                              const isChecked = e.target.checked
                              setSelectedNoteIds((prev) => {
                                if (isChecked) {
                                  if (prev.includes(n.id)) return prev
                                  return [...prev, n.id]
                                }
                                return prev.filter((id) => id !== n.id)
                              })
                            }}
                            style={{ marginRight: 4 }}
                          />
                          <span>{n.name || n.id}</span>
                        </div>
                        <span style={{ fontSize: 11, color: '#6b7280' }}>{statusLabel}</span>
                      </label>
                    )
                  })}
                  {selectedNoteIds.length === 0 && readyNotes.length > 0 && (
                    <div style={{ marginTop: 4, fontSize: 11, color: '#6b7280' }}>
                      Select one or more notes to ground the tutor.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          <div style={{ marginTop: 12, fontSize: 12, color: '#6b7280' }}>
            <div>{activeModeLabel}</div>
            <div>{activeActionLabel}</div>
            <div>{activeTutorActionLabel}</div>
            <div>{activeModelLabel}</div>
          </div>
        </Card>

        {actionMode === 'step' && (
          <Card title="Step controls" style={{ flex: '0 1 260px', minWidth: 240 }}>
            <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6 }}>
              Use these buttons to advance the session without typing.
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {latestStep && Array.isArray(latestStep.controls) && latestStep.controls.length > 0
                ? latestStep.controls.map((ctrl) => {
                    const key = ctrl.type || 'continue'
                    const label =
                      ctrl.label ||
                      (key === 'continue'
                        ? 'Continue'
                        : key === 'skip_to_quiz'
                        ? 'Skip to assessment'
                        : key === 'skip_to_next_concept'
                        ? 'Next concept'
                        : key === 'end_session'
                        ? 'End session'
                        : key)

                    let handler = null
                    if (key === 'continue') {
                      handler = () =>
                        sendToTutor('', {
                          isContinue: true,
                        })
                    } else if (key === 'skip_to_quiz') {
                      handler = () =>
                        sendToTutor('', {
                          isContinue: false,
                          stepControl: { type: 'skip_to_quiz' },
                        })
                    } else if (key === 'skip_to_next_concept') {
                      handler = () =>
                        sendToTutor('', {
                          isContinue: false,
                          stepControl: { type: 'skip_to_next_concept' },
                        })
                    } else if (key === 'end_session') {
                      handler = () =>
                        sendToTutor('', {
                          isContinue: false,
                          stepControl: { type: 'end_session' },
                        })
                    }

                    if (!handler) return null
                    return (
                      <SecondaryButton key={key} onClick={handler} disabled={loading}>
                        {label}
                      </SecondaryButton>
                    )
                  })
                : (
                  <SecondaryButton
                    onClick={() =>
                      sendToTutor('', {
                        isContinue: true,
                      })
                    }
                    disabled={loading}
                  >
                    Continue
                  </SecondaryButton>
                )}
            </div>
          </Card>
        )}

        <Card
          title="Conversation"
          style={{ flex: '2 1 340px', minWidth: 300 }}
        >
          <div
            style={{
              borderRadius: 8,
              border: '1px solid #e5e7eb',
              padding: 12,
              maxHeight: 360,
              overflow: 'auto',
              background: '#f9fafb',
              marginBottom: 12,
            }}
          >
            {turns.length === 0 && (
              <div style={{ fontSize: 13, color: '#6b7280' }}>
                Start the conversation with a question or ask what you should revise today.
              </div>
            )}
            {turns.map((t) => (
              <div
                key={t.id}
                style={{
                  marginBottom: 10,
                  paddingBottom: 8,
                  borderBottom: '1px solid #e5e7eb',
                }}
              >
                <div style={{ fontSize: 12, fontWeight: 600, color: t.role === 'user' ? '#111827' : '#047857' }}>
                  {t.role === 'user' ? 'You' : 'Tutor'}
                </div>
                <div style={{ fontSize: 14, marginTop: 2, whiteSpace: 'pre-wrap' }}>{t.content}</div>
                {t.meta && (
                  <div style={{ marginTop: 4, fontSize: 11, color: '#6b7280' }}>
                    {t.meta.quizPhase === 'quiz' && (
                      <div style={{ marginBottom: 2 }}>
                        <span
                          style={{
                            display: 'inline-block',
                            padding: '1px 6px',
                            borderRadius: 9999,
                            background: '#ecfdf3',
                            border: '1px solid #22c55e',
                            color: '#166534',
                            fontSize: 10,
                            fontWeight: 600,
                            textTransform: 'uppercase',
                            letterSpacing: '0.04em',
                          }}
                        >
                          Quiz Q
                          {typeof t.meta.quizQuestionIndex === 'number' ? t.meta.quizQuestionIndex + 1 : '?'}
                          /
                          {t.meta.quizMaxQuestions || '?'}
                        </span>
                      </div>
                    )}
                    {t.meta.actionType && (
                      <div>
                        Action: <strong>{t.meta.actionType}</strong>
                      </div>
                    )}
                    {(t.meta.concept || t.meta.level) && (
                      <div>
                        Concept: {t.meta.concept || 'n/a'} • Level: {t.meta.level || 'n/a'}
                      </div>
                    )}
                    {Array.isArray(t.meta.sourceChunks) && t.meta.sourceChunks.length > 0 && (
                      <div>
                        Citations:{' '}
                        {t.meta.sourceChunks.map((c) => (
                          <code key={c} style={{ marginRight: 6 }}>
                            {c}
                          </code>
                        ))}
                      </div>
                    )}
                    {t.meta.mcq &&
                      t.meta.actionType === 'ask' &&
                      Array.isArray(t.meta.mcq.options) &&
                      t.meta.mcq.options.length > 0 && (
                        <div style={{ marginTop: 6 }}>
                          <div style={{ marginBottom: 4 }}>Choose an answer:</div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                            {t.meta.mcq.options.map((opt) => (
                              <SecondaryButton
                                key={opt.id}
                                onClick={() =>
                                  void sendToTutor('', {
                                    mcqAnswer: {
                                      question_id: t.meta.mcq.question_id,
                                      option_id: opt.id,
                                    },
                                  })
                                }
                                style={{ textAlign: 'left', justifyContent: 'flex-start' }}
                              >
                                {opt.short_label ? (
                                  <>
                                    <strong>{opt.short_label}.</strong> {opt.label}
                                  </>
                                ) : (
                                  opt.label
                                )}
                              </SecondaryButton>
                            ))}
                          </div>
                        </div>
                      )}
                  </div>
                )}
              </div>
            ))}
            {loading && <div style={{ fontSize: 13, color: '#6b7280' }}>Sending to tutor…</div>}
            <div ref={endRef} />
          </div>

          {error && (
            <div style={{ marginBottom: 8, fontSize: 12, color: '#b91c1c' }}>{error}</div>
          )}

          {actionMode === 'step' && (latestPlan || latestSrlPlan) && (
            <div
              style={{
                marginBottom: 8,
                padding: '10px 12px',
                borderRadius: 8,
                border: '1px solid #bfdbfe',
                background: '#eff6ff',
                fontSize: 12,
                color: '#1f2933',
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 4 }}>Study plan</div>
              {(latestPlan || latestSrlPlan).rationale && (
                <div style={{ marginBottom: 4 }}>{(latestPlan || latestSrlPlan).rationale}</div>
              )}
              {Array.isArray((latestPlan || latestSrlPlan).steps) && (latestPlan || latestSrlPlan).steps.length > 0 && (
                <ol style={{ paddingLeft: 18, margin: 0 }}>
                  {(latestPlan || latestSrlPlan).steps.slice(0, 6).map((step, idx) => {
                    if (!step) return null
                    const label = step.reasoning || step.rationale || ''
                    const action = step.action || 'step'
                    return (
                      <li key={idx} style={{ marginBottom: 2 }}>
                        <strong style={{ textTransform: 'capitalize' }}>{action}</strong>
                        {label ? `  b7 ${label}` : ''}
                      </li>
                    )
                  })}
                </ol>
              )}
            </div>
          )}

          {actionMode === 'step' && (latestStep || latestSrlNextStep) && (
            <div
              style={{
                marginBottom: 8,
                padding: '10px 12px',
                borderRadius: 8,
                border: '1px solid #bfdbfe',
                background: '#eff6ff',
                fontSize: 12,
                color: '#1d4ed8',
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 4 }}>Next planned step:</div>
              <div>
                <strong>Action:</strong>{' '}
                {(latestStep || latestSrlNextStep).action || (latestStep && latestStep.type) || 'auto'}
              </div>
              {(latestStep || latestSrlNextStep).pedagogy_focus &&
                (latestStep || latestSrlNextStep).pedagogy_focus.length > 0 && (
                  <div>
                    <strong>Focus:</strong> {(latestStep || latestSrlNextStep).pedagogy_focus.join(', ')}
                  </div>
                )}
              {((latestStep || latestSrlNextStep).target_concept || (latestStep || latestSrlNextStep).concept) && (
                <div>
                  <strong>Concept:</strong>{' '}
                  {(latestStep || latestSrlNextStep).target_concept || (latestStep || latestSrlNextStep).concept}
                </div>
              )}
            </div>
          )}

          {actionMode === 'step' && latestQuizMeta && latestQuizMeta.quizPhase === 'quiz' && (
            <div
              style={{
                marginBottom: 8,
                padding: '8px 10px',
                borderRadius: 8,
                border: '1px solid #22c55e',
                background: '#ecfdf3',
                fontSize: 12,
                color: '#166534',
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 2 }}>Quiz in progress</div>
              <div>
                Question{' '}
                {Math.min(
                  (typeof latestQuizMeta.quizQuestionIndex === 'number' ? latestQuizMeta.quizQuestionIndex : 0) + 1,
                  typeof latestQuizMeta.quizMaxQuestions === 'number' && latestQuizMeta.quizMaxQuestions > 0
                    ? latestQuizMeta.quizMaxQuestions
                    : Infinity,
                )}{' '}
                of {latestQuizMeta.quizMaxQuestions || '?'} on {latestQuizMeta.concept || 'current concept'}
              </div>
            </div>
          )}

          {multiCandidates && (
            <div
              style={{
                marginBottom: 8,
                padding: '8px 10px',
                borderRadius: 8,
                border: '1px solid #d1d5db',
                background: '#f3f4f6',
                fontSize: 13,
              }}
            >
              <div style={{ marginBottom: 6, fontWeight: 600 }}>Compare candidate answers</div>
              <div style={{ marginBottom: 8, color: '#4b5563' }}>
                Select one answer to adopt. Optional: rate the chosen answer (0 = bad, 1 = okay, 2 = good).
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 8 }}>
                {multiCandidates.candidates.map((c) => (
                  <label
                    key={c.id}
                    style={{
                      borderRadius: 8,
                      border: selectedCandidateId === c.id ? '2px solid #2563eb' : '1px solid #d1d5db',
                      padding: '8px 10px',
                      background: '#ffffff',
                      cursor: 'pointer',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
                      <input
                        type="radio"
                        name="tutor-candidate"
                        value={c.id}
                        checked={selectedCandidateId === c.id}
                        onChange={() => setSelectedCandidateId(c.id)}
                        style={{ marginRight: 6 }}
                      />
                      <span style={{ fontSize: 12, color: '#374151' }}>
                        {c.model_name || c.model_id} • Action: {c.action_type || 'n/a'} • conf:{' '}
                        {c.confidence != null ? Number(c.confidence).toFixed(2) : 'n/a'}
                      </span>
                    </div>
                    <div style={{ fontSize: 14, whiteSpace: 'pre-wrap' }}>{c.response}</div>
                  </label>
                ))}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                {['correctness', 'helpfulness', 'coverage'].map((field) => (
                  <label key={field} style={{ fontSize: 12, color: '#374151' }}>
                    {field.charAt(0).toUpperCase() + field.slice(1)}:{' '}
                    <select
                      value={candidateRating[field] ?? 1}
                      onChange={(e) => handleCandidateRatingChange(field, Number(e.target.value))}
                      style={{ marginLeft: 4, fontSize: 12 }}
                    >
                      <option value={0}>0</option>
                      <option value={1}>1</option>
                      <option value={2}>2</option>
                    </select>
                  </label>
                ))}
              </div>
              <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                <SecondaryButton
                  onClick={() => {
                    setMultiCandidates(null)
                    setSelectedCandidateId(null)
                  }}
                  disabled={loading}
                >
                  Cancel comparison
                </SecondaryButton>
                <PrimaryButton onClick={handleAdoptCandidate} disabled={loading || selectedCandidateId === null}>
                  Adopt answer
                </PrimaryButton>
              </div>
            </div>
          )}

          {actionMode === 'step' && pending && (
            <div
              style={{
                marginBottom: 8,
                padding: '8px 10px',
                borderRadius: 8,
                border: '1px solid #fbbf24',
                background: '#fffbeb',
                fontSize: 12,
                color: '#92400e',
              }}
            >
              Next: &ldquo;{pending}&rdquo;. Confirm to execute this step with the tutor.
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <textarea
              rows={3}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question or propose the next step…"
              style={{
                width: '100%',
                padding: '8px 10px',
                borderRadius: 8,
                border: '1px solid #d1d5db',
                fontSize: 14,
                resize: 'vertical',
              }}
            />
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', flexWrap: 'wrap', alignItems: 'center' }}>
              {actionMode === 'step' && (
                <>
                  <SecondaryButton
                    onClick={() =>
                      void sendToTutor('', {
                        stepControl: { type: 'skip_to_quiz' },
                      })
                    }
                    disabled={loading}
                  >
                    Skip to assessment
                  </SecondaryButton>
                  <SecondaryButton
                    onClick={() =>
                      void sendToTutor('', {
                        stepControl: { type: 'skip_to_next_concept' },
                      })
                    }
                    disabled={loading}
                  >
                    Next concept
                  </SecondaryButton>
                  <SecondaryButton
                    onClick={() => void sendToTutor('', true, 'session_end')}
                    disabled={loading}
                  >
                    End session
                  </SecondaryButton>
                </>
              )}
              {actionMode === 'step' && latestSrlNextStep && !input.trim() && (
                <>
                  <select
                    value={tutorAction}
                    onChange={(e) => setTutorAction(e.target.value)}
                    style={{
                      padding: '7px 10px',
                      borderRadius: 8,
                      border: '1px solid #d1d5db',
                      fontSize: 13,
                      backgroundColor: '#ffffff',
                    }}
                  >
                    <option value="auto">Auto (follow plan)</option>
                    <option value="explain">Explain</option>
                    <option value="ask">Ask</option>
                    <option value="hint">Hint</option>
                    <option value="reflect">Reflect</option>
                    <option value="worked_example">Worked Example</option>
                    <option value="review">Review</option>
                  </select>
                  <PrimaryButton onClick={() => void sendToTutor('', true)} disabled={loading}>
                    Continue
                  </PrimaryButton>
                </>
              )}
              {actionMode === 'step' && input.trim() && (
                <>
                  <SecondaryButton onClick={() => setInput('')} disabled={loading}>
                    Clear
                  </SecondaryButton>
                  <PrimaryButton onClick={() => void sendToTutor(input)} disabled={loading}>
                    Send message
                  </PrimaryButton>
                </>
              )}
              {actionMode === 'step' && !latestSrlNextStep && (
                <PrimaryButton onClick={handleStartStepSession} disabled={loading}>
                  Start session
                </PrimaryButton>
              )}
              {actionMode === 'auto' && (
                <PrimaryButton onClick={() => void sendToTutor(input)} disabled={loading || !input.trim()}>
                  Send
                </PrimaryButton>
              )}
            </div>
          </div>
        </Card>

        {mode === 'notes' && (
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
            onTeachConcept={handleTeachConcept}
            onReviewSelectedConcepts={handleReviewSelectedConcepts}
            onHideConcept={handleHideConcept}
          />
        )}
      </div>
    </main>
  )
}
