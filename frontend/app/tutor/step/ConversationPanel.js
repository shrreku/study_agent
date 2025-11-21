"use client"

import { Card, PrimaryButton, SecondaryButton } from '../../components/ui'

export default function ConversationPanel({
  turns,
  loading,
  input,
  onInputChange,
  onSendMessage,
  onMcqAnswer,
  endRef,
}) {
  const handleSubmit = () => {
    if (!onSendMessage) return
    onSendMessage()
  }

  return (
    <Card title="Conversation" style={{ flex: '2 1 360px', minWidth: 320 }}>
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
            Generate a study plan first. The tutor will then guide you through the steps.
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
                            onClick={() => onMcqAnswer && onMcqAnswer(t.meta, opt)}
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

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <textarea
          rows={3}
          value={input}
          onChange={(e) => onInputChange && onInputChange(e.target.value)}
          placeholder="Ask a question or propose a change to the plan…"
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
          <SecondaryButton onClick={() => onInputChange && onInputChange('')} disabled={loading}>
            Clear
          </SecondaryButton>
          <PrimaryButton onClick={handleSubmit} disabled={loading || !input.trim()}>
            Send message
          </PrimaryButton>
        </div>
      </div>
    </Card>
  )
}
