"use client"

import { useState } from 'react'

const SAMPLE_OBSERVATIONS = `[
  {
    "id": "obs-1",
    "payload": {
      "message": "Can you explain conduction?",
      "user_id": "00000000-0000-0000-0000-000000000201",
      "session_id": "mock-session-201",
      "target_concepts": ["Conduction"],
      "session_policy": {"version": 1, "strategy": "baseline"}
    },
    "observation": {
      "metadata": {"version": 1},
      "user": {"message": "Can you explain conduction?", "user_id": "00000000-0000-0000-0000-000000000201", "target_concepts": ["Conduction"]},
      "classifier": {"intent": "question", "affect": "confused", "concept": "Conduction", "confidence": 0.65, "needs_escalation": false},
      "tutor": {"focus_concept": "Conduction", "concept_level": "beginner", "inference_concept": "Conduction", "learning_path": ["Conduction", "Convection"], "target_concepts": ["Conduction"], "mastery_snapshot": {"mastery": 0.25, "attempts": 0}},
      "retrieval": {"chunk_ids": ["chunk-heat-1"], "source_chunk_ids": ["chunk-heat-1"], "pedagogy_roles": ["definition"], "chunks": [{"id": "chunk-heat-1", "pedagogy_role": "definition", "snippet": "Conduction transfers heat through solids when particles collide."}]},
      "policy": {"cold_start": false, "consecutive_explains": 0, "focus_concept": "Conduction"},
      "session": {"session_id": "mock-session-201", "turn_index": 0, "resource_id": null},
      "action": {"type": "explain", "cold_start": false, "confidence": 0.5, "mastery_delta": null, "source_chunk_ids": ["chunk-heat-1"], "params": {"concept": "Conduction"}}
    }
  },
  {
    "id": "obs-2",
    "payload": {
      "message": "What is convection?",
      "user_id": "00000000-0000-0000-0000-000000000202",
      "session_id": "mock-session-202",
      "target_concepts": ["Convection"],
      "session_policy": {"version": 1, "strategy": "baseline"}
    },
    "observation": {
      "metadata": {"version": 1},
      "user": {"message": "What is convection?", "user_id": "00000000-0000-0000-0000-000000000202", "target_concepts": ["Convection"]},
      "classifier": {"intent": "question", "affect": "engaged", "concept": "Convection", "confidence": 0.7, "needs_escalation": false},
      "tutor": {"focus_concept": "Convection", "concept_level": "beginner", "inference_concept": "Convection", "learning_path": ["Conduction", "Convection", "Radiation"], "target_concepts": ["Convection"], "mastery_snapshot": {"mastery": 0.3, "attempts": 1}},
      "retrieval": {"chunk_ids": ["chunk-heat-2", "chunk-heat-3"], "source_chunk_ids": ["chunk-heat-2"], "pedagogy_roles": ["definition", "example"], "chunks": [{"id": "chunk-heat-2", "pedagogy_role": "definition", "snippet": "Convection moves heat through fluids because warm regions rise."}, {"id": "chunk-heat-3", "pedagogy_role": "example", "snippet": "Boiling water circulates as heat causes water to rise."}]},
      "policy": {"cold_start": false, "consecutive_explains": 1, "focus_concept": "Convection"},
      "session": {"session_id": "mock-session-202", "turn_index": 3, "resource_id": null},
      "action": {"type": "ask", "cold_start": false, "confidence": 0.55, "mastery_delta": null, "source_chunk_ids": ["chunk-heat-2"], "params": {"concept": "Convection"}}
    }
  }
]`

type RolloutResult = {
  sft: any[]
  prefs: any[]
}

export default function RLLabPage() {
  const [observationsText, setObservationsText] = useState<string>(SAMPLE_OBSERVATIONS)
  const [actions, setActions] = useState<string>('explain,ask,hint')
  const [candidates, setCandidates] = useState<number>(2)
  const [promptSet, setPromptSet] = useState<string>('baseline')
  const [mockMode, setMockMode] = useState<boolean>(true)
  const [seed, setSeed] = useState<string>('123')
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [result, setResult] = useState<RolloutResult | null>(null)

  async function runRollout() {
    setLoading(true)
    setError('')
    setResult(null)
    try {
      let parsed
      try {
        parsed = JSON.parse(observationsText)
      } catch (parseErr) {
        throw new Error('Observations JSON is invalid')
      }
      if (!Array.isArray(parsed)) {
        throw new Error('Observations JSON must be an array of objects')
      }
      const body = {
        observations: parsed,
        actions: actions.split(',').map(a => a.trim()).filter(Boolean),
        candidates,
        prompt_set: promptSet || undefined,
        mock: mockMode,
        seed: seed ? Number(seed) : undefined,
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
      setResult(json)
    } catch (err: any) {
      setError(err?.message || String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <main style={{ padding: 24, maxWidth: 1080 }}>
      <div style={{ marginBottom: 16, padding: 12, background: '#e3f2fd', borderRadius: 6, border: '1px solid #90caf9' }}>
        <strong>📌 New Version Available:</strong> Try the improved{' '}
        <a href="/rl-lab-v2" style={{ color: '#1976d2', fontWeight: 600 }}>
          RL Lab v2
        </a>{' '}
        with simplified payload input, step-by-step agent visualization, and cleaner output!
      </div>
      
      <h1>Tutor RL Lab (Original)</h1>
      <p style={{ color: '#444' }}>
        Paste observation payloads, choose candidate actions, and generate mock or live rollouts to inspect SFT and preference datasets.
      </p>

      <section style={{ marginTop: 20, background: '#f7f7ff', padding: 16, border: '1px solid #d7d7f8' }}>
        <h2 style={{ marginTop: 0 }}>1. Observations</h2>
        <p style={{ marginBottom: 8 }}>Provide an array of observation entries (each should include <code>payload</code> and <code>observation</code> blocks).</p>
        <textarea
          value={observationsText}
          onChange={e => setObservationsText(e.target.value)}
          rows={18}
          style={{ width: '100%', fontFamily: 'monospace', fontSize: 14, padding: 12 }}
        />
      </section>

      <section style={{ marginTop: 20, background: '#f5f5f5', padding: 16, border: '1px solid #e0e0e0' }}>
        <h2 style={{ marginTop: 0 }}>2. Rollout Settings</h2>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
          <label style={{ minWidth: 220 }}>
            Actions (comma separated)
            <input type="text" value={actions} onChange={e => setActions(e.target.value)} style={{ width: '100%' }} />
          </label>
          <label>
            Candidates per observation
            <input type="number" min={1} max={8} value={candidates} onChange={e => setCandidates(Number(e.target.value) || 1)} style={{ width: 80 }} />
          </label>
          <label>
            Prompt set
            <input type="text" value={promptSet} onChange={e => setPromptSet(e.target.value)} style={{ width: 160 }} />
          </label>
          <label>
            Seed (optional)
            <input type="number" value={seed} onChange={e => setSeed(e.target.value)} style={{ width: 120 }} />
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input type="checkbox" checked={mockMode} onChange={e => setMockMode(e.target.checked)} />
            Mock mode (deterministic)
          </label>
        </div>
        <button onClick={runRollout} disabled={loading} style={{ marginTop: 16 }}>
          {loading ? 'Generating…' : 'Generate Rollout'}
        </button>
        {error && <div style={{ marginTop: 12, color: '#b00020' }}>{error}</div>}
      </section>

      {result && (
        <section style={{ marginTop: 20, background: '#ffffff', padding: 16, border: '1px solid #dddddd' }}>
          <h2 style={{ marginTop: 0 }}>3. Results</h2>
          <p>
            Generated <strong>{result.sft.length}</strong> SFT records and <strong>{result.prefs.length}</strong> preference pairs.
            Copy the JSON below into your training or evaluation workflows.
          </p>
          <details open style={{ marginTop: 12 }}>
            <summary style={{ cursor: 'pointer', fontWeight: 600 }}>SFT JSONL Preview</summary>
            <pre style={{ background: '#f8f8f8', padding: 12, fontSize: 13 }}>
              {result.sft.map(entry => JSON.stringify(entry)).join('\n')}
            </pre>
          </details>
          <details style={{ marginTop: 12 }}>
            <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Preference JSONL Preview</summary>
            <pre style={{ background: '#f8f8f8', padding: 12, fontSize: 13 }}>
              {result.prefs.map(entry => JSON.stringify(entry)).join('\n')}
            </pre>
          </details>
        </section>
      )}
    </main>
  )
}

