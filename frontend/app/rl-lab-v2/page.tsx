"use client"

import { useState } from 'react'

const SAMPLE_PAYLOAD = `{
  "message": "Can you explain conduction?",
  "user_id": "11111111-2222-3333-4444-555555555555",
  "target_concepts": ["Conduction"]
}`

type AgentStep = {
  step: string
  title: string
  data: any
  description: string
}

type CandidateConfig = {
  id: string
  action: string
  model: string
}

type CandidateResult = {
  action_type: string
  response: string
  reward: any
  critic: any
  observation?: any
  steps?: AgentStep[]
  model?: string
  configured_action?: string
  srl_plan?: any
  srl_critique?: any
}

export default function RLLabV2Page() {
  const [payloadText, setPayloadText] = useState<string>(SAMPLE_PAYLOAD)
  const [candidates, setCandidates] = useState<CandidateConfig[]>([
    { id: '1', action: 'auto', model: 'anthropic/claude-haiku-4.5' },
    { id: '2', action: 'auto', model: 'anthropic/claude-haiku-4.5' },
  ])
  const [promptSet, setPromptSet] = useState<string>('baseline')
  const [mockMode, setMockMode] = useState<boolean>(true)
  const [seed, setSeed] = useState<string>('123')
  const [criticModel, setCriticModel] = useState<string>('openai/gpt-5-mini-08-07')
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [results, setResults] = useState<CandidateResult[]>([])
  const [expandedSteps, setExpandedSteps] = useState<Record<number, boolean>>({})
  const [expandedObservation, setExpandedObservation] = useState<Record<number, boolean>>({})

  const addCandidate = () => {
    setCandidates([
      ...candidates,
      { id: String(candidates.length + 1), action: 'auto', model: 'gpt-4o-mini' },
    ])
  }

  const removeCandidate = (id: string) => {
    if (candidates.length > 1) {
      setCandidates(candidates.filter(c => c.id !== id))
    }
  }

  const updateCandidate = (id: string, field: 'action' | 'model', value: string) => {
    setCandidates(candidates.map(c => c.id === id ? { ...c, [field]: value } : c))
  }

  async function runRollout() {
    setLoading(true)
    setError('')
    setResults([])
    setExpandedSteps({})
    setExpandedObservation({})
    
    try {
      let payload
      try {
        payload = JSON.parse(payloadText)
      } catch (parseErr) {
        throw new Error('Payload JSON is invalid')
      }
      
      // Wrap payload in observation format
      const observation = {
        payload: payload,
      }
      
      const body = {
        observations: [observation],
        actions: candidates.map(c => c.action),
        candidates: candidates.length,
        prompt_set: promptSet || undefined,
        mock: mockMode,
        seed: seed ? Number(seed) : undefined,
        simplified: false,  // Get full format to extract data properly
        detailed_steps: true,
        model_per_candidate: candidates.map(c => ({ action: c.action, model: c.model })),
        critic_model: criticModel || undefined,
      }
      
      const response = await fetch('http://localhost:8000/api/rl/rollout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-token',
        },
        body: JSON.stringify(body),
      })
      
      if (!response.ok) {
        const text = await response.text()
        throw new Error(`HTTP ${response.status}: ${text}`)
      }
      
      const json = await response.json()
      
      // Combine SFT records with steps and observation
      const sftRecords = json.sft || []
      const stepsData = json.steps || []
      
      const combined: CandidateResult[] = sftRecords.map((sft: any, idx: number) => {
        const fullReward = sft.reward || {}
        const components = fullReward.components || {}

        const observation = sft.observation || {}
        const srl = observation.srl || {}
        const srlPlan = srl.plan || null
        const srlCritique = srl.critique || null

        const baseSteps: AgentStep[] = (stepsData[idx]?.steps || []) as AgentStep[]
        const enrichedSteps: AgentStep[] = []

        if (srlPlan) {
          enrichedSteps.push({
            step: 'srl_plan',
            title: 'SRL Plan',
            description: `Planned ${srlPlan.intended_action || 'action'} with confidence ${(srlPlan.confidence ?? 0).toFixed(2)}`,
            data: srlPlan,
          })
        }

        if (srlCritique) {
          enrichedSteps.push({
            step: 'srl_critique',
            title: 'SRL Self-Critique',
            description: `Critique quality ${(srlCritique.quality ?? 0).toFixed(2)}${
              srlCritique.should_revise === true
                ? ' (suggests revision)'
                : srlCritique.should_revise === false
                ? ' (no revision suggested)'
                : ''
            }`,
            data: srlCritique,
          })
        }

        const mergedSteps: AgentStep[] = [...enrichedSteps, ...baseSteps]
        
        // Get the actual action type from observation or sft data
        const actualActionType = observation.action?.type || sft.action?.type || sft.action_type || 'explain'
        
        return {
          action_type: actualActionType,
          response: sft.response || '',
          reward: {
            rubric: components.rubric?.score || 0.0,
            intent: components.intent?.score || 0.0,
            gating: components.gating?.score || 0.0,
            grounding: components.grounding?.score || 0.0,
            style: components.style?.score || 0.0,
            total: fullReward.total || 0.0,
            flags: fullReward.flags || [],
          },
          critic: sft.critic || {},
          observation,
          steps: mergedSteps,
          model: candidates[idx]?.model || 'gpt-4o-mini',
          configured_action: candidates[idx]?.action,
          srl_plan: srlPlan,
          srl_critique: srlCritique,
        }
      })
      
      setResults(combined)
    } catch (err: any) {
      setError(err?.message || String(err))
    } finally {
      setLoading(false)
    }
  }

  const toggleSteps = (idx: number) => {
    setExpandedSteps(prev => ({ ...prev, [idx]: !prev[idx] }))
  }

  const toggleObservation = (idx: number) => {
    setExpandedObservation(prev => ({ ...prev, [idx]: !prev[idx] }))
  }

  return (
    <main style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h1 style={{ marginBottom: 8 }}>Tutor RL Lab v2</h1>
      <p style={{ color: '#666', marginBottom: 24 }}>
        Send a minimal payload, configure candidates with different LLM models, and compare responses
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
        {/* Input Section */}
        <section style={{ background: '#f7f7ff', padding: 20, borderRadius: 8, border: '1px solid #d7d7f8' }}>
          <h2 style={{ marginTop: 0, fontSize: 18 }}>1. Student Message</h2>
          <p style={{ fontSize: 14, color: '#666', marginBottom: 12 }}>
            Provide a minimal payload with message, user_id, and target_concepts
          </p>
          <textarea
            value={payloadText}
            onChange={e => setPayloadText(e.target.value)}
            rows={12}
            style={{ 
              width: '100%', 
              fontFamily: 'monospace', 
              fontSize: 13, 
              padding: 12,
              borderRadius: 4,
              border: '1px solid #ccc',
            }}
          />
        </section>

        {/* Settings Section */}
        <section style={{ background: '#f5f5f5', padding: 20, borderRadius: 8, border: '1px solid #e0e0e0' }}>
          <h2 style={{ marginTop: 0, fontSize: 18 }}>2. Rollout Settings</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontSize: 14, fontWeight: 500 }}>Prompt Set</span>
                <input 
                  type="text" 
                  value={promptSet} 
                  onChange={e => setPromptSet(e.target.value)} 
                  style={{ padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
                />
              </label>
              
              <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontSize: 14, fontWeight: 500 }}>Seed</span>
                <input 
                  type="number" 
                  value={seed} 
                  onChange={e => setSeed(e.target.value)} 
                  style={{ padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
                />
              </label>
            </div>
            
            <div style={{ marginTop: 12 }}>
              <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontSize: 14, fontWeight: 500 }}>Critic Model</span>
                <input 
                  type="text" 
                  value={criticModel} 
                  onChange={e => setCriticModel(e.target.value)} 
                  placeholder="gpt-4o-mini"
                  style={{ padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
                />
                <span style={{ fontSize: 12, color: '#666' }}>Model used for critic scoring (applies to all candidates)</span>
              </label>
            </div>
            
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, marginTop: 12 }}>
              <input 
                type="checkbox" 
                checked={mockMode} 
                onChange={e => setMockMode(e.target.checked)} 
              />
              Mock mode (deterministic, no LLM/DB required)
            </label>
          </div>
          
          <div style={{ marginTop: 20, paddingTop: 20, borderTop: '1px solid #d0d0d0' }}>
            <h3 style={{ marginTop: 0, fontSize: 16, marginBottom: 12 }}>3. Configure Candidates</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {candidates.map((candidate, idx) => (
                <div key={candidate.id} style={{ 
                  padding: 12, 
                  background: 'white', 
                  borderRadius: 6, 
                  border: '1px solid #ddd',
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr auto',
                  gap: 10,
                  alignItems: 'end',
                }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <label style={{ fontSize: 12, fontWeight: 500 }}>Action</label>
                    <select 
                      value={candidate.action}
                      onChange={e => updateCandidate(candidate.id, 'action', e.target.value)}
                      style={{ padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
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
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <label style={{ fontSize: 12, fontWeight: 500 }}>LLM Model</label>
                    <input
                      type="text" 
                      value={candidate.model}
                      onChange={e => updateCandidate(candidate.id, 'model', e.target.value)}
                      placeholder="Enter AIMLAPI model ID"
                      style={{ padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
                    />
                  </div>
                  
                  <button
                    onClick={() => removeCandidate(candidate.id)}
                    disabled={candidates.length === 1}
                    style={{
                      padding: '8px 12px',
                      background: '#f44336',
                      color: 'white',
                      border: 'none',
                      borderRadius: 4,
                      cursor: candidates.length === 1 ? 'not-allowed' : 'pointer',
                      opacity: candidates.length === 1 ? 0.5 : 1,
                      fontSize: 12,
                    }}
                  >
                    Remove
                  </button>
                </div>
              ))}
              
              <button
                onClick={addCandidate}
                style={{
                  padding: '10px 16px',
                  background: '#2196F3',
                  color: 'white',
                  border: 'none',
                  borderRadius: 4,
                  cursor: 'pointer',
                  fontWeight: 500,
                  fontSize: 14,
                }}
              >
                + Add Candidate
              </button>
            </div>
          </div>
          
          <button 
            onClick={runRollout} 
            disabled={loading}
            style={{ 
              marginTop: 20,
              width: '100%',
              padding: '12px 20px',
              fontSize: 16,
              fontWeight: 600,
              background: loading ? '#ccc' : '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              cursor: loading ? 'not-allowed' : 'pointer',
            }}
          >
            {loading ? 'Generating…' : 'Generate Rollout'}
          </button>
          
          {error && (
            <div style={{ marginTop: 12, padding: 12, background: '#fee', color: '#c00', borderRadius: 4, fontSize: 14 }}>
              {error}
            </div>
          )}
        </section>
      </div>

      {/* Results Section */}
      {results.length > 0 && (
        <section style={{ marginTop: 32 }}>
          <h2 style={{ fontSize: 20, marginBottom: 16 }}>Results ({results.length} candidates)</h2>
          
          {results.map((result, idx) => (
            <div 
              key={idx} 
              style={{ 
                marginBottom: 24, 
                background: 'white', 
                border: '2px solid #e0e0e0',
                borderRadius: 8,
                overflow: 'hidden',
              }}
            >
              {/* Candidate Header */}
              <div style={{ 
                padding: '16px 20px', 
                background: '#f9f9f9', 
                borderBottom: '1px solid #e0e0e0',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}>
                <div>
                  <h3 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>
                    Candidate {idx + 1}: <span style={{ color: '#2196F3' }}>{result.action_type}</span>
                    {result.configured_action === 'auto' && (
                      <span style={{ fontSize: 11, color: '#666', marginLeft: 8, fontStyle: 'italic' }}>
                        (agent decided)
                      </span>
                    )}
                    {result.model && <span style={{ fontSize: 12, color: '#666', marginLeft: 12 }}>Model: {result.model}</span>}
                  </h3>
                </div>
                <div style={{ display: 'flex', gap: 16, fontSize: 14 }}>
                  <span title="Total Reward">
                    ⭐ {(result.reward?.total || 0).toFixed(2)}
                  </span>
                  <span title="Critic Confidence">
                    🎯 {(result.critic?.confidence || 0).toFixed(2)}
                  </span>
                </div>
              </div>

              {/* Response */}
              <div style={{ padding: 20 }}>
                <h4 style={{ marginTop: 0, marginBottom: 12, fontSize: 15, color: '#555' }}>Response:</h4>
                <div style={{ 
                  padding: 16, 
                  background: '#fafafa', 
                  borderRadius: 6,
                  borderLeft: '4px solid #2196F3',
                  fontSize: 14,
                  lineHeight: 1.6,
                  whiteSpace: 'pre-wrap',
                  maxHeight: 300,
                  overflow: 'auto',
                }}>
                  {result.response || '(empty response)'}
                </div>
              </div>

              {/* Observation & Action */}
              {result.observation && Object.keys(result.observation).length > 0 && (
                <div style={{ padding: '0 20px 20px' }}>
                  <details style={{ cursor: 'pointer' }}>
                    <summary style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: '#555' }}>
                      📋 Agent Observation & Decision
                    </summary>
                    <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                      {/* Classification */}
                      <div style={{ padding: 12, background: '#e3f2fd', borderRadius: 6 }}>
                        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: '#0369a1' }}>Classification</div>
                        <div style={{ fontSize: 13 }}>
                          <div>Intent: {result.observation.classifier?.intent || 'N/A'}</div>
                          <div>Affect: {result.observation.classifier?.affect || 'N/A'}</div>
                          <div>Concept: {result.observation.classifier?.concept || 'N/A'}</div>
                          <div>Confidence: {(result.observation.classifier?.confidence || 0).toFixed(2)}</div>
                        </div>
                      </div>

                      {/* Tutor Decision */}
                      <div style={{ padding: 12, background: '#f3e5f5', borderRadius: 6 }}>
                        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: '#6a1b9a' }}>Tutor Decision</div>
                        <div style={{ fontSize: 13 }}>
                          <div>Focus Concept: {result.observation.tutor?.focus_concept || 'N/A'}</div>
                          <div>Level: {result.observation.tutor?.concept_level || 'N/A'}</div>
                          <div>Learning Path: {(result.observation.tutor?.learning_path || []).join(', ') || 'N/A'}</div>
                          <div>Chunks Retrieved: {(result.observation.retrieval?.chunk_ids || []).length || 0}</div>
                        </div>
                      </div>
                    </div>

                    {/* Full JSON (collapsible) */}
                    <details style={{ marginTop: 12 }}>
                      <summary style={{ fontSize: 12, color: '#666', cursor: 'pointer' }}>View Full Observation JSON</summary>
                      <pre style={{ 
                        marginTop: 8, 
                        padding: 12, 
                        background: '#f9f9f9', 
                        borderRadius: 4,
                        overflow: 'auto',
                        fontSize: 11,
                        maxHeight: 300,
                      }}>
                        {JSON.stringify(result.observation, null, 2)}
                      </pre>
                    </details>
                  </details>
                </div>
              )}

              {/* Scores */}
              <div style={{ padding: '0 20px 20px' }}>
                <details style={{ cursor: 'pointer' }}>
                  <summary style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: '#555' }}>
                    📊 Detailed Scores
                  </summary>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginTop: 12 }}>
                    {/* Reward Components */}
                    <div style={{ padding: 12, background: '#f0f9ff', borderRadius: 6 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: '#0369a1' }}>Reward Components</div>
                      <div style={{ fontSize: 13 }}>
                        <div>Rubric: {(result.reward?.rubric || 0).toFixed(2)}</div>
                        <div>Intent: {(result.reward?.intent || 0).toFixed(2)}</div>
                        <div>Gating: {(result.reward?.gating || 0).toFixed(2)}</div>
                        <div>Grounding: {(result.reward?.grounding || 0).toFixed(2)}</div>
                        <div>Style: {(result.reward?.style || 0).toFixed(2)}</div>
                        <div style={{ marginTop: 8, fontWeight: 600, color: '#0369a1' }}>Total: {(result.reward?.total || 0).toFixed(2)}</div>
                      </div>
                    </div>

                    {/* Critic Scores */}
                    <div style={{ padding: 12, background: '#fef3c7', borderRadius: 6 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: '#92400e' }}>Critic Scores</div>
                      <div style={{ fontSize: 13 }}>
                        <div>Clarity: {(result.critic?.clarity || 0).toFixed(2)}</div>
                        <div>Accuracy: {(result.critic?.accuracy || 0).toFixed(2)}</div>
                        <div>Support: {(result.critic?.support || 0).toFixed(2)}</div>
                        <div>Confidence: {(result.critic?.confidence || 0).toFixed(2)}</div>
                        <div>Hallucination: {result.critic?.hallucination ? '⚠️ Yes' : '✅ No'}</div>
                      </div>
                    </div>

                    {/* SRL Metrics */}
                    <div style={{ padding: 12, background: '#ecfdf3', borderRadius: 6 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: '#166534' }}>SRL Metrics</div>
                      <div style={{ fontSize: 13 }}>
                        <div>Planned Action: {result.srl_plan?.intended_action || 'N/A'}</div>
                        <div>Plan Confidence: {((result.srl_plan?.confidence ?? 0) as number).toFixed(2)}</div>
                        <div>Plan Steps: {Array.isArray(result.srl_plan?.steps) ? result.srl_plan.steps.length : 0}</div>
                        <div>
                          Critique Quality: {((result.srl_critique?.quality ?? 0) as number).toFixed(2)}
                        </div>
                        <div>
                          Should Revise:{' '}
                          {result.srl_critique?.should_revise === true
                            ? 'Yes'
                            : result.srl_critique?.should_revise === false
                            ? 'No'
                            : 'N/A'}
                        </div>
                      </div>
                    </div>

                    {/* Flags */}
                    <div style={{ padding: 12, background: '#fef2f2', borderRadius: 6 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: '#991b1b' }}>Flags</div>
                      <div style={{ fontSize: 13 }}>
                        {(result.reward?.flags || []).length > 0 ? (
                          (result.reward.flags as string[]).map((flag: string, fi: number) => (
                            <div key={fi}>⚠️ {flag}</div>
                          ))
                        ) : (
                          <div style={{ color: '#16a34a' }}>✅ No flags</div>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  {result.critic?.notes && (
                    <div style={{ marginTop: 12, padding: 12, background: '#f9fafb', borderRadius: 6, fontSize: 13 }}>
                      <strong>Critic Notes:</strong> {result.critic.notes}
                    </div>
                  )}
                </details>
              </div>

              {/* Agent Steps */}
              {result.steps && result.steps.length > 0 && (
                <div style={{ padding: '0 20px 20px' }}>
                  <details style={{ cursor: 'pointer' }}>
                    <summary style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: '#555' }}>
                      🔍 Agent Steps ({result.steps.length})
                    </summary>
                    <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
                      {result.steps.map((step: AgentStep, si: number) => (
                        <div 
                          key={si} 
                          style={{ 
                            padding: 16, 
                            background: '#fafafa', 
                            borderRadius: 6,
                            borderLeft: '3px solid #9333ea',
                          }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                            <strong style={{ fontSize: 14, color: '#9333ea' }}>
                              Step {si + 1}: {step.title}
                            </strong>
                          </div>
                          <div style={{ fontSize: 13, color: '#666', marginBottom: 8 }}>
                            {step.description}
                          </div>
                          <details style={{ fontSize: 12 }}>
                            <summary style={{ cursor: 'pointer', color: '#666' }}>View data</summary>
                            <pre style={{ 
                              marginTop: 8, 
                              padding: 12, 
                              background: 'white', 
                              borderRadius: 4,
                              overflow: 'auto',
                              fontSize: 11,
                              maxHeight: 200,
                            }}>
                              {JSON.stringify(step.data, null, 2)}
                            </pre>
                          </details>
                        </div>
                      ))}
                    </div>
                  </details>
                </div>
              )}
            </div>
          ))}
        </section>
      )}
    </main>
  )
}
