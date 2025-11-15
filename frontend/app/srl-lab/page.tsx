'use client'

import React from 'react'

type TutorResponse = any

function FieldLabel({ children }: React.PropsWithChildren<{}>) {
  return <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>{children}</div>
}

function Section({ title, children }: React.PropsWithChildren<{ title: string }>) {
  return (
    <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 12, marginBottom: 16 }}>
      <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>{title}</div>
      {children}
    </div>
  )
}

function KV({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', gap: 8, padding: '2px 0' }}>
      <div style={{ width: 160, color: '#374151' }}>{k}</div>
      <div style={{ flex: 1 }}>{v as any}</div>
    </div>
  )
}

function Textarea({ value, onChange, rows = 4 }: { value: string; onChange: (s: string) => void; rows?: number }) {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      rows={rows}
      style={{ width: '100%', border: '1px solid #e5e7eb', borderRadius: 6, padding: 8 }}
    />
  )
}

function Input({ value, onChange, placeholder }: { value: string; onChange: (s: string) => void; placeholder?: string }) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      style={{ width: '100%', border: '1px solid #e5e7eb', borderRadius: 6, padding: 8 }}
    />
  )
}

export default function SRLLabPage() {
  const [backendUrl, setBackendUrl] = React.useState<string>('http://localhost:8000')
  const [token, setToken] = React.useState<string>('dev')
  const [message, setMessage] = React.useState<string>('Can you explain conduction?')
  const [userId, setUserId] = React.useState<string>('00000000-0000-0000-0000-000000000201')
  const [targetConcepts, setTargetConcepts] = React.useState<string>('Conduction')
  const [resourceId, setResourceId] = React.useState<string>('')
  const [sessionId, setSessionId] = React.useState<string>('')
  const [emitState, setEmitState] = React.useState<boolean>(true)
  const [loading, setLoading] = React.useState<boolean>(false)
  const [error, setError] = React.useState<string>('')
  const [results, setResults] = React.useState<TutorResponse[]>([])
  const [candidates, setCandidates] = React.useState<Array<{ id: string; override: string; model: string }>>([
    { id: '1', override: 'auto', model: '' },
  ])

  const canSend = message.trim().length > 0 && userId.trim().length > 0 && backendUrl.trim().length > 0 && token.trim().length > 0

  async function runCandidates() {
    setLoading(true)
    setError('')
    setResults([])
    try {
      const basePayload: any = {
        message: message.trim(),
        user_id: userId.trim(),
        target_concepts: targetConcepts
          .split(',')
          .map((s) => s.trim())
          .filter((s) => s.length > 0),
        emit_state: emitState,
        session_policy: { version: 1, strategy: 'baseline' },
      }
      if (resourceId.trim()) basePayload.resource_id = resourceId.trim()
      if (sessionId.trim()) basePayload.session_id = sessionId.trim()

      const endpoint = `${backendUrl.replace(/\/$/, '')}/api/agent/tutor`
      const reqs = candidates.map(async (cand) => {
        const payload = { ...basePayload }
        if (cand.model.trim()) (payload as any).model_hint = cand.model.trim()
        if (cand.override !== 'auto') (payload as any).action_override = { type: cand.override }
        const resp = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
          body: JSON.stringify(payload),
        })
        if (!resp.ok) {
          const text = await resp.text()
          throw new Error(text || `HTTP ${resp.status}`)
        }
        const data = await resp.json()
        return { data, cand }
      })
      const settled = await Promise.all(reqs)
      setResults(settled.map((x) => ({ ...x.data, _candidate: x.cand })))
    } catch (e: any) {
      setError(e?.message || 'request_failed')
    } finally {
      setLoading(false)
    }
  }

  function addCandidate() {
    setCandidates((prev) => [...prev, { id: String(prev.length + 1), override: 'auto', model: '' }])
  }
  function removeCandidate(id: string) {
    setCandidates((prev) => (prev.length > 1 ? prev.filter((c) => c.id !== id) : prev))
  }
  function updateCandidate(id: string, field: 'override' | 'model', value: string) {
    setCandidates((prev) => prev.map((c) => (c.id === id ? { ...c, [field]: value } : c)))
  }

  function Badge({ children, color = '#3b82f6' }: React.PropsWithChildren<{ color?: string }>) {
    return (
      <span
        style={{
          display: 'inline-block',
          fontSize: 12,
          borderRadius: 999,
          padding: '2px 8px',
          background: color,
          color: 'white',
          marginRight: 6,
        }}
      >
        {children}
      </span>
    )
  }

  // Per-candidate details are computed inside the results map below

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 16 }}>
      <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 12 }}>SRL Lab</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 16 }}>
        <div>
          <Section title="Backend">
            <KV k="Base URL" v={<Input value={backendUrl} onChange={setBackendUrl} />} />
            <KV k="Auth Token" v={<Input value={token} onChange={setToken} />} />
          </Section>
          <Section title="Session">
            <KV k="User ID" v={<Input value={userId} onChange={setUserId} />} />
            <KV k="Session ID" v={<Input value={sessionId} onChange={setSessionId} />} />
            <KV k="Resource ID" v={<Input value={resourceId} onChange={setResourceId} />} />
          </Section>
          <Section title="Prompt">
            <FieldLabel>Message</FieldLabel>
            <Textarea value={message} onChange={setMessage} rows={5} />
            <KV k="Targets" v={<Input value={targetConcepts} onChange={setTargetConcepts} />} />
            <KV k="Emit State" v={<input type="checkbox" checked={emitState} onChange={(e) => setEmitState(e.target.checked)} />} />
          </Section>
          <Section title="Candidates">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {candidates.map((c) => (
                <div key={c.id} style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 8, display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: 8, alignItems: 'end' }}>
                  <div>
                    <FieldLabel>Override</FieldLabel>
                    <select value={c.override} onChange={(e) => updateCandidate(c.id, 'override', e.target.value)} style={{ width: '100%', padding: 8, border: '1px solid #e5e7eb', borderRadius: 6 }}>
                      <option value="auto">auto (agent decides)</option>
                      <option value="explain">explain</option>
                      <option value="ask">ask</option>
                      <option value="hint">hint</option>
                      <option value="reflect">reflect</option>
                      <option value="worked_example">worked_example</option>
                      <option value="review">review</option>
                    </select>
                  </div>
                  <div>
                    <FieldLabel>Model Hint</FieldLabel>
                    <Input value={c.model} onChange={(v) => updateCandidate(c.id, 'model', v)} placeholder="e.g. anthropic/claude-haiku-4.5" />
                  </div>
                  <button onClick={() => removeCandidate(c.id)} disabled={candidates.length === 1} style={{ padding: '8px 12px', background: '#ef4444', color: 'white', borderRadius: 6, cursor: candidates.length === 1 ? 'not-allowed' : 'pointer', opacity: candidates.length === 1 ? 0.5 : 1 }}>Remove</button>
                </div>
              ))}
              <div>
                <button onClick={addCandidate} style={{ padding: '8px 12px', background: '#2563eb', color: 'white', borderRadius: 6 }}>+ Add Candidate</button>
              </div>
              <div>
                <button onClick={runCandidates} disabled={!canSend || loading} style={{ marginTop: 4, padding: '10px 16px', background: '#111827', color: 'white', borderRadius: 6, width: '100%' }}>
                  {loading ? 'Running…' : 'Generate Candidates'}
                </button>
              </div>
              {!canSend && <div style={{ color: '#ef4444', marginTop: 8, fontSize: 12 }}>Set backend URL, token, message, and user id.</div>}
              {error && <div style={{ color: '#ef4444', marginTop: 8, fontSize: 12 }}>{error}</div>}
            </div>
          </Section>
        </div>
        <div>
          {results.length === 0 ? (
            <Section title="Results">
              <div>Run candidates to see results.</div>
            </Section>
          ) : (
            results.map((result: any, idx: number) => {
              const observation = (result && result.observation) || null
              const srlFromObservation = observation && observation.srl
              const srlFromResponsePlan = result && (result as any).srl_plan
              const srlFromResponseCrit = result && (result as any).srl_critique
              const progress = (result && (result as any).progress) || []
              const progressSteps = Array.isArray(progress) ? progress.filter((p: any) => p && p.stage === 'step') : []
              const obsChunks = (observation && observation.retrieval && Array.isArray(observation.retrieval.chunks)) ? observation.retrieval.chunks : []
              const resultSourceIds = (result && Array.isArray((result as any).source_chunk_ids)) ? (result as any).source_chunk_ids : []
              const configured = (result && (result as any)._candidate) || { override: 'auto', model: '' }

              return (
                <div key={idx} style={{ border: '2px solid #e5e7eb', borderRadius: 10, marginBottom: 16, overflow: 'hidden' }}>
                  <div style={{ padding: '12px 16px', background: '#f9fafb', borderBottom: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ fontWeight: 800 }}>Candidate {idx + 1}: <span style={{ color: '#2563eb' }}>{result.action_type || 'n/a'}</span> {configured.override === 'auto' && <span style={{ fontSize: 12, color: '#6b7280' }}>(agent decided)</span>}</div>
                    <div style={{ display: 'flex', gap: 12, fontSize: 13 }}>
                      <span title="Confidence">🎯 {typeof result.confidence === 'number' ? result.confidence.toFixed(3) : 'n/a'}</span>
                      {configured.model && (<span title="Model" style={{ color: '#6b7280' }}>Model: {configured.model}</span>)}
                    </div>
                  </div>

                  <div style={{ padding: 12 }}>
                    <Section title="Result Summary">
                      <div>
                        <div style={{ marginBottom: 8 }}>
                          <Badge color="#111827">{result.action_type || 'n/a'}</Badge>
                          <Badge>{typeof result.confidence === 'number' ? result.confidence.toFixed(3) : 'n/a'}</Badge>
                        </div>
                        <div style={{ whiteSpace: 'pre-wrap' }}>{result.response || ''}</div>
                        {Array.isArray(resultSourceIds) && resultSourceIds.length > 0 && (
                          <>
                            <FieldLabel>Sources</FieldLabel>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                              {resultSourceIds.map((sid: string, i: number) => (
                                <span key={sid || i} style={{ fontSize: 11, padding: '2px 6px', background: '#e5e7eb', borderRadius: 999 }}>{String(sid)}</span>
                              ))}
                            </div>
                          </>
                        )}
                      </div>
                    </Section>

                    {observation && (
                      <Section title="Classifier">
                        <KV k="Intent" v={observation.classifier?.intent} />
                        <KV k="Affect" v={observation.classifier?.affect} />
                        <KV k="Concept" v={observation.classifier?.concept} />
                        <KV k="Confidence" v={String(observation.classifier?.confidence ?? '')} />
                      </Section>
                    )}

                    {(srlFromObservation || srlFromResponsePlan) && (
                      <Section title="SRL Plan">
                        {srlFromObservation?.plan ? (
                          <>
                            <KV k="Intended Action" v={srlFromObservation.plan.intended_action} />
                            <KV k="Rationale" v={srlFromObservation.plan.rationale} />
                            <KV k="Confidence" v={String(srlFromObservation.plan.confidence)} />
                            <FieldLabel>Thinking</FieldLabel>
                            <div style={{ whiteSpace: 'pre-wrap', background: '#f9fafb', padding: 8, borderRadius: 6 }}>{srlFromObservation.plan.thinking || ''}</div>
                            <FieldLabel>Assumptions</FieldLabel>
                            <div>{Array.isArray(srlFromObservation.plan.assumptions) ? srlFromObservation.plan.assumptions.join(', ') : ''}</div>
                            <FieldLabel>Risks</FieldLabel>
                            <div>{Array.isArray(srlFromObservation.plan.risks) ? srlFromObservation.plan.risks.join(', ') : ''}</div>
                            {Array.isArray(srlFromObservation.plan.target_sequence) && srlFromObservation.plan.target_sequence.length > 0 && (
                              <>
                                <FieldLabel>Target Sequence</FieldLabel>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                                  {srlFromObservation.plan.target_sequence.map((t: string, i: number) => (
                                    <span key={i} style={{ fontSize: 11, padding: '2px 6px', background: '#dcfce7', borderRadius: 999 }}>{t}</span>
                                  ))}
                                </div>
                              </>
                            )}
                            {Array.isArray(srlFromObservation.plan.steps) && srlFromObservation.plan.steps.length > 0 && (
                              <>
                                <FieldLabel>Steps</FieldLabel>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                  {srlFromObservation.plan.steps.map((st: any, i: number) => (
                                    <div key={i} style={{ background: '#eef2ff', padding: 8, borderRadius: 6 }}>
                                      <div style={{ fontWeight: 700 }}>{st.action || 'step'}</div>
                                      <div style={{ fontSize: 12 }}>{st.rationale || ''}</div>
                                      {Array.isArray(st.pedagogy_focus) && st.pedagogy_focus.length > 0 && (
                                        <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                                          {st.pedagogy_focus.map((r: string, j: number) => (
                                            <span key={j} style={{ fontSize: 11, padding: '2px 6px', background: '#dbeafe', borderRadius: 999 }}>{r}</span>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </>
                            )}
                          </>
                        ) : srlFromResponsePlan ? (
                          <>
                            <KV k="Rationale" v={srlFromResponsePlan.rationale} />
                            <KV k="Confidence" v={String(srlFromResponsePlan.confidence)} />
                            <FieldLabel>Thinking</FieldLabel>
                            <div style={{ whiteSpace: 'pre-wrap', background: '#f9fafb', padding: 8, borderRadius: 6 }}>{srlFromResponsePlan.thinking || ''}</div>
                            {Array.isArray(srlFromResponsePlan.target_sequence) && srlFromResponsePlan.target_sequence.length > 0 && (
                              <>
                                <FieldLabel>Target Sequence</FieldLabel>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                                  {srlFromResponsePlan.target_sequence.map((t: string, i: number) => (
                                    <span key={i} style={{ fontSize: 11, padding: '2px 6px', background: '#dcfce7', borderRadius: 999 }}>{t}</span>
                                  ))}
                                </div>
                              </>
                            )}
                            {Array.isArray(srlFromResponsePlan.steps) && srlFromResponsePlan.steps.length > 0 && (
                              <>
                                <FieldLabel>Steps</FieldLabel>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                  {srlFromResponsePlan.steps.map((st: any, i: number) => (
                                    <div key={i} style={{ background: '#eef2ff', padding: 8, borderRadius: 6 }}>
                                      <div style={{ fontWeight: 700 }}>{st.action || 'step'}</div>
                                      <div style={{ fontSize: 12 }}>{st.rationale || ''}</div>
                                      {Array.isArray(st.pedagogy_focus) && st.pedagogy_focus.length > 0 && (
                                        <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                                          {st.pedagogy_focus.map((r: string, j: number) => (
                                            <span key={j} style={{ fontSize: 11, padding: '2px 6px', background: '#dbeafe', borderRadius: 999 }}>{r}</span>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </>
                            )}
                          </>
                        ) : null}
                      </Section>
                    )}

                    {(srlFromObservation?.critique || srlFromResponseCrit) && (
                      <Section title="SRL Critique">
                        {srlFromObservation?.critique ? (
                          <>
                            <KV k="Quality" v={String(srlFromObservation.critique.quality)} />
                            <KV k="Should Revise" v={String(srlFromObservation.critique.should_revise)} />
                            <FieldLabel>Issues</FieldLabel>
                            <div>{Array.isArray(srlFromObservation.critique.issues) ? srlFromObservation.critique.issues.join(', ') : ''}</div>
                            <FieldLabel>Suggestions</FieldLabel>
                            <div>{Array.isArray(srlFromObservation.critique.suggestions) ? srlFromObservation.critique.suggestions.join(', ') : ''}</div>
                          </>
                        ) : srlFromResponseCrit ? (
                          <>
                            <KV k="Quality" v={String(srlFromResponseCrit.quality)} />
                            <KV k="Should Revise" v={String(srlFromResponseCrit.should_revise)} />
                            <FieldLabel>Issues</FieldLabel>
                            <div>{Array.isArray(srlFromResponseCrit.issues) ? srlFromResponseCrit.issues.join(', ') : ''}</div>
                            <FieldLabel>Suggestions</FieldLabel>
                            <div>{Array.isArray(srlFromResponseCrit.suggestions) ? srlFromResponseCrit.suggestions.join(', ') : ''}</div>
                          </>
                        ) : null}
                      </Section>
                    )}

                    {observation && (
                      <Section title="Retrieval">
                        <KV k="Query" v={String(observation.retrieval?.query || '')} />
                        <KV k="Pedagogy Roles" v={(observation.retrieval?.pedagogy_roles || []).join(', ')} />
                        <KV k="Chunk IDs" v={(observation.retrieval?.chunk_ids || []).join(', ')} />
                        <KV k="Source IDs" v={(observation.retrieval?.source_chunk_ids || []).join(', ')} />
                        {Array.isArray(observation.retrieval?.chunks) && observation.retrieval.chunks.length > 0 && (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                            {observation.retrieval.chunks.map((c: any, cidx: number) => (
                              <div key={c.id || cidx} style={{ border: '1px solid #e5e7eb', borderRadius: 6, padding: 8 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                  <div style={{ fontWeight: 700 }}>{String(c.id || '').slice(0, 8)}</div>
                                  <div style={{ display: 'flex', gap: 6 }}>
                                    {c.pedagogy_role && <span style={{ fontSize: 11, padding: '2px 6px', background: '#fef3c7', borderRadius: 999 }}>{c.pedagogy_role}</span>}
                                    {c.page_number != null && <span style={{ fontSize: 11, padding: '2px 6px', background: '#e5e7eb', borderRadius: 999 }}>p{String(c.page_number)}</span>}
                                  </div>
                                </div>
                                <div style={{ marginTop: 6, fontSize: 12, color: '#374151' }}>
                                  score: {String(c.score ?? '')} • sim: {String(c.sim ?? '')} • bm25: {String(c.bm25 ?? '')}
                                </div>
                                {c.snippet && (
                                  <div style={{ marginTop: 6, whiteSpace: 'pre-wrap', background: '#f9fafb', padding: 8, borderRadius: 6 }}>{c.snippet}</div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </Section>
                    )}

                    {Array.isArray(progressSteps) && progressSteps.length > 0 && (
                      <Section title="Plan Execution (Steps)">
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                          {progressSteps.map((p: any, pidx: number) => (
                            <div key={pidx} style={{ padding: 8, background: '#ecfeff', borderRadius: 6 }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div style={{ fontWeight: 700 }}>Step {typeof p.index === 'number' ? p.index + 1 : pidx + 1} • {p.action || 'step'}</div>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                                  {Array.isArray(p.roles) && p.roles.map((r: string, i: number) => (
                                    <span key={i} style={{ fontSize: 11, padding: '2px 6px', background: '#bae6fd', borderRadius: 999 }}>{r}</span>
                                  ))}
                                </div>
                              </div>
                              <div style={{ marginTop: 6, fontSize: 12 }}>
                                <div><strong>Query:</strong> {String(p.query || '')}</div>
                                <div><strong>Retrieved:</strong> {String(p.retrieved ?? '')}</div>
                                {Array.isArray(p.chunk_ids) && p.chunk_ids.length > 0 && (
                                  <div style={{ marginTop: 4 }}>
                                    <FieldLabel>Chunk IDs</FieldLabel>
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                      {p.chunk_ids.map((cid: string, j: number) => (
                                        <span key={cid || j} style={{ fontSize: 11, padding: '2px 6px', background: '#e5e7eb', borderRadius: 999 }}>{String(cid)}</span>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                              {(() => {
                                const ids = Array.isArray(p.chunk_ids) ? p.chunk_ids : []
                                const details = obsChunks.filter((c: any) => ids.includes(c.id))
                                if (!details.length) return null
                                return (
                                  <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
                                    {details.map((c: any, j: number) => (
                                      <div key={c.id || j} style={{ border: '1px dashed #93c5fd', borderRadius: 6, padding: 8 }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                          <div style={{ fontWeight: 700 }}>{String(c.id || '').slice(0, 8)}</div>
                                          <div style={{ display: 'flex', gap: 6 }}>
                                            {c.pedagogy_role && <span style={{ fontSize: 11, padding: '2px 6px', background: '#fef3c7', borderRadius: 999 }}>{c.pedagogy_role}</span>}
                                            {c.page_number != null && <span style={{ fontSize: 11, padding: '2px 6px', background: '#e5e7eb', borderRadius: 999 }}>p{String(c.page_number)}</span>}
                                          </div>
                                        </div>
                                        <div style={{ marginTop: 6, fontSize: 12, color: '#374151' }}>
                                          score: {String(c.score ?? '')} • sim: {String(c.sim ?? '')} • bm25: {String(c.bm25 ?? '')}
                                        </div>
                                        {c.snippet && (
                                          <div style={{ marginTop: 6, whiteSpace: 'pre-wrap', background: '#f9fafb', padding: 8, borderRadius: 6 }}>{c.snippet}</div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                )
                              })()}
                            </div>
                          ))}
                        </div>
                      </Section>
                    )}

                    {Array.isArray(progress) && progress.length > 0 && (
                      <Section title="Progress (Stages)">
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                          {progress.map((p: any, sidx: number) => (
                            <div key={sidx} style={{ padding: 8, background: '#f3f4f6', borderRadius: 6 }}>
                              <div style={{ fontWeight: 700, marginBottom: 4 }}>{p.stage || 'stage'}</div>
                              <div style={{ fontSize: 12, whiteSpace: 'pre-wrap' }}>{JSON.stringify(p, null, 2)}</div>
                            </div>
                          ))}
                        </div>
                      </Section>
                    )}

                    {observation && (
                      <Section title="Action">
                        <KV k="Type" v={observation.action?.type} />
                        <KV k="Confidence" v={String(observation.action?.confidence ?? '')} />
                        <KV k="Cold Start" v={String(observation.action?.cold_start ?? '')} />
                        <KV k="Override Type" v={String(observation.action?.override_type ?? '')} />
                        <KV k="Applied Override Type" v={String(observation.action?.applied_override_type ?? '')} />
                        <FieldLabel>Params</FieldLabel>
                        <div style={{ whiteSpace: 'pre-wrap', background: '#f9fafb', padding: 8, borderRadius: 6 }}>{JSON.stringify(observation.action?.params || {}, null, 2)}</div>
                      </Section>
                    )}

                    <Section title="Raw JSON">
                      <div style={{ fontSize: 12, whiteSpace: 'pre-wrap', background: '#f9fafb', padding: 8, borderRadius: 6 }}>{JSON.stringify(result, null, 2)}</div>
                    </Section>
                  </div>
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}
