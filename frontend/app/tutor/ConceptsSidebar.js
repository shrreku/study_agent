import { Card, PrimaryButton, SecondaryButton } from '../components/ui'

export default function ConceptsSidebar({
  readyNotes,
  selectedNoteIds,
  concepts,
  conceptsLoading,
  conceptsError,
  conceptFeedbackError,
  visibleConcepts,
  conceptFilter,
  onChangeConceptFilter,
  conceptSort,
  onChangeConceptSort,
  showHiddenConcepts,
  onChangeShowHidden,
  selectedConcepts,
  loading,
  latestTutorConcept,
  masteryPillStyle,
  onToggleConceptSelection,
  onTeachConcept,
  onReviewSelectedConcepts,
  onHideConcept,
}) {
  const hasConcepts = !conceptsError && Array.isArray(concepts) && concepts.length > 0

  return (
    <Card
      title="Concepts in selected notes"
      style={{ flex: '1 1 260px', minWidth: 260 }}
    >
      {readyNotes.length === 0 && (
        <p style={{ fontSize: 13, color: '#6b7280' }}>
          No ready notes found. Upload notes on the <a href="/notes">Notes</a> page to see concepts here.
        </p>
      )}

      {readyNotes.length > 0 && selectedNoteIds.length === 0 && (
        <p style={{ fontSize: 13, color: '#6b7280' }}>
          Select one or more notes on the left to view the concepts they contain.
        </p>
      )}

      {conceptsError && (
        <div style={{ marginTop: 6, fontSize: 12, color: '#b91c1c' }}>{conceptsError}</div>
      )}

      {conceptFeedbackError && !conceptsError && (
        <div style={{ marginTop: 4, fontSize: 11, color: '#b91c1c' }}>{conceptFeedbackError}</div>
      )}

      {!conceptsError && conceptsLoading && (
        <div style={{ marginTop: 6, fontSize: 12, color: '#6b7280' }}>Loading concepts…</div>
      )}

      {!conceptsError && !conceptsLoading && selectedNoteIds.length > 0 && concepts.length === 0 && (
        <div style={{ marginTop: 6, fontSize: 12, color: '#6b7280' }}>
          No concepts found yet for the selected notes.
        </div>
      )}

      {hasConcepts && (
        <>
          <div
            style={{
              marginTop: 6,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 8,
              flexWrap: 'wrap',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <label style={{ fontSize: 11, color: '#4b5563' }}>
                Filter:
                <select
                  value={conceptFilter}
                  onChange={(e) => onChangeConceptFilter(e.target.value)}
                  style={{
                    marginLeft: 4,
                    fontSize: 11,
                    borderRadius: 9999,
                    padding: '2px 6px',
                    border: '1px solid #d1d5db',
                  }}
                >
                  <option value="all">All</option>
                  <option value="weak">Weak / Unknown</option>
                </select>
              </label>
              <label style={{ fontSize: 11, color: '#4b5563' }}>
                Sort:
                <select
                  value={conceptSort}
                  onChange={(e) => onChangeConceptSort(e.target.value)}
                  style={{
                    marginLeft: 4,
                    fontSize: 11,
                    borderRadius: 9999,
                    padding: '2px 6px',
                    border: '1px solid #d1d5db',
                  }}
                >
                  <option value="weak_first">Weakest first</option>
                  <option value="path">Learning path</option>
                  <option value="alpha">A–Z</option>
                </select>
              </label>
              <label style={{ fontSize: 11, color: '#4b5563' }}>
                <input
                  type="checkbox"
                  checked={!!showHiddenConcepts}
                  onChange={(e) => onChangeShowHidden && onChangeShowHidden(e.target.checked)}
                  style={{ marginRight: 4 }}
                />
                Show hidden
              </label>
            </div>
            <div style={{ fontSize: 11, color: '#6b7280' }}>{visibleConcepts.length} concepts</div>
          </div>

          <div
            style={{
              marginTop: 8,
              display: 'flex',
              flexDirection: 'column',
              gap: 6,
              maxHeight: 260,
              overflow: 'auto',
            }}
          >
            {visibleConcepts.map((c) => {
              const selected = Array.isArray(selectedConcepts) && selectedConcepts.includes(c.concept)

              return (
                <div
                  key={`${c.canonical}-${c.level}-${c.occurrences}`}
                  style={{
                    borderRadius: 8,
                    border: '1px solid #e5e7eb',
                    background: '#ffffff',
                    padding: '6px 8px',
                    fontSize: 12,
                    opacity: c.hidden ? 0.55 : 1,
                  }}
                >
                  <div style={{ flex: '1 1 auto' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ fontSize: 13, fontWeight: 500, color: '#111827' }}>{c.concept}</div>
                      {onHideConcept && (
                        <button
                          type="button"
                          onClick={() => onHideConcept(c)}
                          style={{
                            border: 'none',
                            background: 'transparent',
                            padding: 0,
                            fontSize: 11,
                            color: '#b91c1c',
                            cursor: 'pointer',
                          }}
                        >
                          Hide
                        </button>
                      )}
                    </div>
                    <div style={{ marginTop: 2, fontSize: 11, color: '#6b7280' }}>
                      {Array.isArray(c.resource_ids) ? `${c.resource_ids.length} notes` : ''}
                      {Array.isArray(c.pages) && c.pages.length > 0 &&
                        ` \u2022 ${c.pages.length} page${c.pages.length > 1 ? 's' : ''}`}
                    </div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
                    <span style={masteryPillStyle(c.level)}>
                      {c.level || 'Unknown'}
                      {c.mastery != null && ` (${Number(c.mastery).toFixed(2)})`}
                    </span>
                    <label style={{ fontSize: 11, color: '#4b5563', display: 'flex', alignItems: 'center' }}>
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={() => onToggleConceptSelection(c.concept)}
                        style={{ marginRight: 4 }}
                      />
                      Select
                    </label>
                    <PrimaryButton
                      onClick={() => onTeachConcept(c.concept)}
                      style={{ padding: '4px 8px', fontSize: 11 }}
                      disabled={loading}
                    >
                      Teach
                    </PrimaryButton>
                  </div>
                </div>
              )
            })}
          </div>

          {selectedConcepts.length > 0 && (
            <div style={{ marginTop: 8, display: 'flex', justifyContent: 'flex-end' }}>
              <SecondaryButton onClick={onReviewSelectedConcepts} disabled={loading}>
                Teach selected concepts
              </SecondaryButton>
            </div>
          )}
        </>
      )}
    </Card>
  )
}
