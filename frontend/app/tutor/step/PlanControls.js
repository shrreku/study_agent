"use client"

import { Card, PrimaryButton, SecondaryButton } from '../../components/ui'

export default function PlanControls({
  readyNotes,
  selectedNoteIds,
  onChangeSelectedNoteIds,
  models,
  selectedModelId,
  onChangeSelectedModelId,
  strategy,
  onChangeStrategy,
  highLevelPlan,
  latestSrlPlan,
  latestSrlNextStep,
  latestPlan,
  latestStep,
  planApproved,
  loading,
  error,
  nextActionOverride,
  onChangeNextActionOverride,
  onGeneratePlan,
  onApprovePlan,
  onContinue,
  onSkipToQuiz,
  onNextConcept,
  onEndSession,
}) {
  const handleToggleNote = (noteId, checked) => {
    if (!onChangeSelectedNoteIds) return
    onChangeSelectedNoteIds((prev) => {
      const prevArr = Array.isArray(prev) ? prev : []
      if (checked) {
        if (prevArr.includes(noteId)) return prevArr
        return [...prevArr, noteId]
      }
      return prevArr.filter((id) => id !== noteId)
    })
  }

  const effectivePlan = latestPlan || latestSrlPlan
  const effectiveNextStep = latestStep || latestSrlNextStep

  const hasNextStep = !!effectiveNextStep
  const hasHighLevelPlan = typeof highLevelPlan === 'string' && highLevelPlan.trim().length > 0

  return (
    <Card title="Plan & controls" style={{ flex: '2 1 360px', minWidth: 320 }}>
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#6b7280' }}>Notes</div>
        {readyNotes.length === 0 ? (
          <p style={{ marginTop: 6, fontSize: 13, color: '#6b7280' }}>
            No ready notes found. Upload a PDF or slides on the <a href="/notes">Notes</a> page, then refresh this page.
          </p>
        ) : (
          <div
            style={{
              marginTop: 6,
              maxHeight: 140,
              overflow: 'auto',
              borderRadius: 8,
              border: '1px solid #d1d5db',
              padding: 8,
            }}
          >
            {readyNotes.map((n) => {
              const checked = selectedNoteIds.includes(n.id)
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
                      onChange={(e) => handleToggleNote(n.id, e.target.checked)}
                      style={{ marginRight: 4 }}
                    />
                    <span>{n.name || n.id}</span>
                  </div>
                  {/* Status label is not critical here and requires mapJobStatusToLabel; omit for simplicity */}
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

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#6b7280' }}>Model</div>
        <select
          value={selectedModelId}
          onChange={(e) => onChangeSelectedModelId && onChangeSelectedModelId(e.target.value)}
          style={{
            marginTop: 6,
            width: '100%',
            padding: '7px 10px',
            borderRadius: 8,
            border: '1px solid #d1d5db',
            fontSize: 14,
          }}
        >
          {(models || []).map((m) => (
            <option key={m.id} value={m.id}>
              {m.display_name || m.id}
            </option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#6b7280' }}>
          Concept strategy
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 6, flexWrap: 'wrap' }}>
          <SecondaryButton
            onClick={() => onChangeStrategy && onChangeStrategy('learning_path')}
            style={
              strategy === 'learning_path'
                ? { background: '#2563eb0d', borderColor: '#2563eb', color: '#1d4ed8' }
                : null
            }
          >
            Follow learning path
          </SecondaryButton>
          <SecondaryButton
            onClick={() => onChangeStrategy && onChangeStrategy('weakest_first')}
            style={
              strategy === 'weakest_first'
                ? { background: '#2563eb0d', borderColor: '#2563eb', color: '#1d4ed8' }
                : null
            }
          >
            Weakest first
          </SecondaryButton>
          <SecondaryButton
            onClick={() => onChangeStrategy && onChangeStrategy('pick')}
            style={
              strategy === 'pick' ? { background: '#2563eb0d', borderColor: '#2563eb', color: '#1d4ed8' } : null
            }
          >
            I choose concepts
          </SecondaryButton>
        </div>
        <p style={{ marginTop: 6, fontSize: 12, color: '#6b7280' }}>
          This choice controls which concepts the planner targets. You can always re-generate the plan with a different
          strategy.
        </p>
      </div>

      <div
        style={{
          marginBottom: 12,
          display: 'flex',
          gap: 8,
          flexWrap: 'wrap',
          alignItems: 'center',
        }}
      >
        {planApproved && hasNextStep && (
          <>
            <span style={{ fontSize: 12, color: '#6b7280' }}>Next action:</span>
            <select
              value={nextActionOverride}
              onChange={(e) => onChangeNextActionOverride && onChangeNextActionOverride(e.target.value)}
              style={{
                padding: '7px 10px',
                borderRadius: 8,
                border: '1px solid #d1d5db',
                fontSize: 13,
                backgroundColor: '#ffffff',
              }}
              disabled={loading}
            >
              <option value="auto">Auto (follow plan)</option>
              <option value="explain">Explain</option>
              <option value="ask">Ask</option>
              <option value="hint">Hint</option>
              <option value="reflect">Reflect</option>
              <option value="worked_example">Worked example</option>
              <option value="review">Review</option>
            </select>
          </>
        )}

        {(() => {
          let label = 'Generate study plan'
          let onClick = onGeneratePlan
          let disabled = loading || readyNotes.length === 0

          // Once a high-level plan exists but is not yet approved, switch
          // the primary CTA to approval, which also kicks off SRL.
          if (hasHighLevelPlan && !planApproved) {
            label = 'Approve plan & start session'
            onClick = onApprovePlan
            disabled = loading
          } else if (planApproved && hasNextStep) {
            label = 'Continue'
            onClick = onContinue
            disabled = loading
          }

          return (
            <PrimaryButton onClick={onClick} disabled={disabled}>
              {label}
            </PrimaryButton>
          )
        })()}
      </div>

      {error && (
        <div style={{ marginBottom: 8, fontSize: 12, color: '#b91c1c' }}>{error}</div>
      )}

      {hasHighLevelPlan && (
        <div
          style={{
            marginBottom: 8,
            padding: '10px 12px',
            borderRadius: 8,
            border: '1px solid #e5e7eb',
            background: '#f9fafb',
            fontSize: 12,
            color: '#111827',
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 4 }}>High-level study plan (pinned)</div>
          <div style={{ whiteSpace: 'pre-wrap' }}>{highLevelPlan}</div>
        </div>
      )}

      {effectivePlan && (
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
          <div style={{ fontWeight: 600, marginBottom: 4 }}>SRL plan for current concept</div>
          {effectivePlan.rationale && <div style={{ marginBottom: 4 }}>{effectivePlan.rationale}</div>}
          {Array.isArray(effectivePlan.steps) && effectivePlan.steps.length > 0 && (
            <ol style={{ paddingLeft: 18, margin: 0 }}>
              {effectivePlan.steps.slice(0, 6).map((step, idx) => {
                if (!step) return null
                const label = step.reasoning || step.rationale || ''
                const action = step.action || 'step'
                return (
                  <li key={idx} style={{ marginBottom: 2 }}>
                    <strong style={{ textTransform: 'capitalize' }}>{action}</strong>
                    {label ? ` — ${label}` : ''}
                  </li>
                )
              })}
            </ol>
          )}
        </div>
      )}

      {effectiveNextStep && (
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
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Next planned step</div>
          <div>
            <strong>Action:</strong> {effectiveNextStep.action || effectiveNextStep.type || 'auto'}
          </div>
          {effectiveNextStep.pedagogy_focus && effectiveNextStep.pedagogy_focus.length > 0 && (
            <div>
              <strong>Focus:</strong> {effectiveNextStep.pedagogy_focus.join(', ')}
            </div>
          )}
          {(effectiveNextStep.target_concept || effectiveNextStep.concept) && (
            <div>
              <strong>Concept:</strong> {effectiveNextStep.target_concept || effectiveNextStep.concept}
            </div>
          )}
        </div>
      )}

      <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {latestStep && Array.isArray(latestStep.controls) && latestStep.controls.length > 0
          ? latestStep.controls.map((ctrl) => {
              const key = ctrl.type || 'continue'
              const label = ctrl.label ||
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
              let disabled = loading
              if (key === 'continue') {
                handler = onContinue
                disabled = loading || !planApproved
              } else if (key === 'skip_to_quiz') {
                handler = onSkipToQuiz
                disabled = loading || !planApproved
              } else if (key === 'skip_to_next_concept') {
                handler = onNextConcept
                disabled = loading || !planApproved
              } else if (key === 'end_session') {
                handler = onEndSession
              }

              if (!handler) return null
              return (
                <SecondaryButton key={key} onClick={handler} disabled={disabled}>
                  {label}
                </SecondaryButton>
              )
            })
          : (
            <>
              <SecondaryButton onClick={onSkipToQuiz} disabled={loading || !planApproved}>
                Skip to assessment
              </SecondaryButton>
              <SecondaryButton onClick={onNextConcept} disabled={loading || !planApproved}>
                Next concept
              </SecondaryButton>
              <SecondaryButton onClick={onEndSession} disabled={loading}>
                End session
              </SecondaryButton>
            </>
          )}
      </div>
    </Card>
  )
}
