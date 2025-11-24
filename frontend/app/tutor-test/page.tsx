'use client'

import { useState } from 'react'

interface ConceptPlan {
    plan_id: string
    concept_id: string
    steps: Array<{
        step_type: string
        instruction: string
        subgoal: string
        pedagogical_action?: string
        estimated_duration: number
    }>
    total_steps: number
}

interface InitResponse {
    session_id: string
    concept_id: string
    concept_plan: ConceptPlan
    mastery_map: Record<string, number>
    target_concepts: string[]
    concept_metadata: {
        display_name: string
        description: string
        target_mastery: number
    }
}

export default function TutorTestPage() {
    const [sessionId, setSessionId] = useState('')
    const [loading, setLoading] = useState(false)
    const [debugView, setDebugView] = useState(true)
    const [conceptPlan, setConceptPlan] = useState<ConceptPlan | null>(null)
    const [masteryMap, setMasteryMap] = useState<Record<string, number>>({})
    const [conceptMetadata, setConceptMetadata] = useState<any>(null)
    const [lastResponse, setLastResponse] = useState<any>(null)
    const [lastDebug, setLastDebug] = useState<any>(null)

    const initializeTestSession = async () => {
        setLoading(true)
        try {
            const res = await fetch('/api/test/init-tutor-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer test-token',
                },
            })

            if (!res.ok) {
                throw new Error(`Error: ${res.status} ${res.statusText}`)
            }

            const data: InitResponse = await res.json()

            setSessionId(data.session_id)
            setConceptPlan(data.concept_plan)
            setMasteryMap(data.mastery_map)
            setConceptMetadata(data.concept_metadata)
            setLastResponse({
                content: `✓ Test session initialized!\n\nConcept: ${data.concept_metadata.display_name}\n${data.concept_metadata.description}\n\nExecuting first action automatically...`
            })

            // Auto-execute first action
            setTimeout(() => {
                sendControlWithSession(data.session_id, {
                    message: '',
                    confirmed_action: 'continue'
                })
            }, 500)
        } catch (err: any) {
            console.error(err)
            alert(`Failed to initialize: ${err.message}`)
        } finally {
            setLoading(false)
        }
    }

    const sendControlWithSession = async (sid: string, payload: any) => {
        setLoading(true)

        try {
            const body = {
                message: payload.message || '',
                ...payload
            }

            console.log('[DEBUG] Sending payload:', JSON.stringify(body, null, 2))

            const res = await fetch(`/api/tutor/pedagogy/session/${sid}/message`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer test-token',
                },
                body: JSON.stringify(body),
            })

            if (!res.ok) {
                const errorText = await res.text()
                throw new Error(`Error: ${res.status} ${res.statusText}\n${errorText}`)
            }

            const data = await res.json()

            const responseContent = data.messages
                ? data.messages.map((m: any) => m.content).join('\n\n')
                : JSON.stringify(data)

            setLastResponse({
                content: responseContent,
                action: payload.confirmed_action || payload.step_control?.type || 'action'
            })
            setLastDebug(data)
        } catch (err: any) {
            console.error(err)
            setLastResponse({
                content: `Error: ${err.message}`,
                error: true
            })
        } finally {
            setLoading(false)
        }
    }

    const sendControl = async (payload: any) => {
        if (!sessionId) {
            alert('Please initialize a session first')
            return
        }
        await sendControlWithSession(sessionId, payload)
    }

    const handleNextStep = () => sendControl({ message: '', confirmed_action: 'continue' })
    const handleReplan = () => sendControl({ message: '', step_control: { type: 'replan_concept' } })
    const handleNextConcept = () => sendControl({ message: '', step_control: { type: 'skip_to_next_concept' } })

    // Extract current step info
    const currentAction = lastDebug?.debug?.pedagogical_action || 'N/A'
    const planIndex = lastDebug?.debug?.plan_index ?? 0
    const planLength = conceptPlan?.total_steps || lastDebug?.debug?.plan_length || 0
    const currentMastery = lastDebug?.debug?.mastery ?? masteryMap[conceptMetadata?.concept_id || 'test_concept'] ?? 0
    const targetMastery = conceptMetadata?.target_mastery ?? 0.8
    const phase = lastDebug?.debug?.phase || 'learning'

    // Determine next action from concept plan
    const nextAction = conceptPlan && planIndex < conceptPlan.steps.length
        ? conceptPlan.steps[planIndex]?.pedagogical_action || conceptPlan.steps[planIndex]?.step_type
        : 'End of plan'

    return (
        <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '20px' }}>
            <h1 style={{ fontSize: '28px', fontWeight: 'bold', marginBottom: '20px' }}>
                Tutor MDP Test Environment
            </h1>

            {/* Session Controls */}
            <div style={{ marginBottom: '20px', display: 'flex', gap: '10px', alignItems: 'center' }}>
                <button
                    onClick={initializeTestSession}
                    disabled={loading}
                    style={{
                        padding: '10px 20px',
                        background: '#10b981',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: loading ? 'not-allowed' : 'pointer',
                        fontWeight: 'bold',
                    }}
                >
                    {loading ? 'Initializing...' : 'Initialize Test Session'}
                </button>
                <div style={{ flex: 1 }} />
                <button
                    onClick={() => setDebugView(!debugView)}
                    style={{
                        padding: '8px 16px',
                        background: '#6b7280',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                    }}
                >
                    {debugView ? 'Hide Debug' : 'Show Debug'}
                </button>
            </div>

            {sessionId && (
                <div style={{ marginBottom: '10px', fontSize: '14px', color: '#6b7280' }}>
                    Session ID: <span style={{ fontFamily: 'monospace' }}>{sessionId}</span>
                </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
                {/* Left Column: Controls and Response */}
                <div>
                    {/* Control Panel */}
                    {sessionId && (
                        <div style={{
                            marginBottom: '20px',
                            padding: '16px',
                            background: '#fef3c7',
                            borderRadius: '8px',
                            border: '1px solid #fde68a'
                        }}>
                            <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '12px', color: '#92400e' }}>
                                MDP Controls
                            </div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                                <button
                                    onClick={handleNextStep}
                                    disabled={loading}
                                    style={{
                                        padding: '12px 20px',
                                        background: '#3b82f6',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '6px',
                                        cursor: loading ? 'not-allowed' : 'pointer',
                                        fontWeight: 'bold',
                                        fontSize: '14px',
                                    }}
                                >
                                    ▶ Next Step
                                </button>
                                <button
                                    onClick={handleReplan}
                                    disabled={loading}
                                    style={{
                                        padding: '12px 20px',
                                        background: '#f59e0b',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '6px',
                                        cursor: loading ? 'not-allowed' : 'pointer',
                                        fontWeight: 'bold',
                                        fontSize: '14px',
                                    }}
                                >
                                    🔄 Re-plan
                                </button>
                                <button
                                    onClick={handleNextConcept}
                                    disabled={loading}
                                    style={{
                                        padding: '12px 20px',
                                        background: '#ef4444',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '6px',
                                        cursor: loading ? 'not-allowed' : 'pointer',
                                        fontWeight: 'bold',
                                        fontSize: '14px',
                                    }}
                                >
                                    ⏭ Next Concept / End
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Tutor Response */}
                    {lastResponse && (
                        <div style={{
                            padding: '16px',
                            background: lastResponse.error ? '#fee2e2' : '#f0f9ff',
                            borderRadius: '8px',
                            border: lastResponse.error ? '1px solid #fecaca' : '1px solid #bae6fd',
                        }}>
                            <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '8px', color: '#1e40af' }}>
                                {lastResponse.action ? `Action: ${lastResponse.action}` : 'Tutor Response'}
                            </div>
                            <div style={{ fontSize: '14px', whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
                                {lastResponse.content}
                            </div>
                        </div>
                    )}

                    {/* Debug Info */}
                    {debugView && lastDebug && (
                        <div style={{ marginTop: '20px' }}>
                            <div style={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '8px', color: '#6b7280' }}>
                                Debug Info
                            </div>
                            <pre style={{
                                fontSize: '10px',
                                padding: '12px',
                                background: '#f8f9fa',
                                borderRadius: '4px',
                                overflowX: 'auto',
                                maxHeight: '300px',
                                overflowY: 'auto',
                                border: '1px solid #e5e7eb',
                            }}>
                                {JSON.stringify(lastDebug, null, 2)}
                            </pre>
                        </div>
                    )}
                </div>

                {/* Right Column: State and Plan */}
                <div>
                    {/* MDP State Display */}
                    {sessionId && conceptPlan && (
                        <div style={{
                            marginBottom: '20px',
                            padding: '16px',
                            background: '#f3f4f6',
                            borderRadius: '8px',
                            border: '1px solid #d1d5db'
                        }}>
                            <div style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '12px' }}>
                                MDP State
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '14px' }}>
                                <div>
                                    <strong>Current Action:</strong> <span style={{ color: '#059669' }}>{currentAction}</span>
                                </div>
                                <div>
                                    <strong>Next Action:</strong> <span style={{ color: '#3b82f6' }}>{nextAction}</span>
                                </div>
                                <div>
                                    <strong>Progress:</strong> {planIndex + 1} / {planLength}
                                </div>
                                <div>
                                    <strong>Phase:</strong> {phase}
                                </div>
                                <div>
                                    <strong>Current Mastery:</strong> {(currentMastery * 100).toFixed(1)}%
                                </div>
                                <div>
                                    <strong>Target Mastery:</strong> {(targetMastery * 100).toFixed(1)}%
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Concept Plan Steps */}
                    {sessionId && conceptPlan && (
                        <div style={{
                            padding: '16px',
                            background: '#eff6ff',
                            borderRadius: '8px',
                            border: '1px solid #bfdbfe'
                        }}>
                            <div style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '4px' }}>
                                Concept: {conceptMetadata?.display_name || conceptPlan.concept_id}
                            </div>
                            <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '12px' }}>
                                Plan ID: {conceptPlan.plan_id} • {conceptPlan.total_steps} steps
                            </div>
                            <div style={{ fontSize: '12px', maxHeight: '500px', overflowY: 'auto' }}>
                                {conceptPlan.steps.map((step, idx) => {
                                    const actionType = step.pedagogical_action || step.step_type.toUpperCase()
                                    return (
                                        <div
                                            key={idx}
                                            style={{
                                                padding: '10px',
                                                marginBottom: '6px',
                                                background: idx === planIndex ? '#dbeafe' : 'white',
                                                borderRadius: '4px',
                                                border: idx === planIndex ? '2px solid #3b82f6' : '1px solid #e5e7eb',
                                            }}
                                        >
                                            <div style={{ fontWeight: idx === planIndex ? 'bold' : 'normal', marginBottom: '4px', color: '#1e40af' }}>
                                                {idx + 1}. [{actionType}]
                                            </div>
                                            <div style={{ fontSize: '11px', fontWeight: 'bold', marginBottom: '2px' }}>
                                                {step.subgoal}
                                            </div>
                                            <div style={{ fontSize: '10px', color: '#6b7280' }}>
                                                {step.instruction}
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
