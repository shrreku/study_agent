const NOTES_STORAGE_KEY = 'studyagent_notes'

export function loadNotesMetadata() {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(NOTES_STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function saveNotesMetadata(notes) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(NOTES_STORAGE_KEY, JSON.stringify(notes))
  } catch {
    // no-op
  }
}

export function upsertNoteMetadata(nextNote) {
  if (!nextNote || !nextNote.id) return
  const notes = loadNotesMetadata()
  const idx = notes.findIndex((n) => n.id === nextNote.id)
  if (idx === -1) {
    notes.unshift(nextNote)
  } else {
    notes[idx] = { ...notes[idx], ...nextNote }
  }
  saveNotesMetadata(notes)
}

export function mapJobStatusToLabel(status) {
  if (!status) return 'In progress'
  const s = String(status).toLowerCase()
  if (s.includes('queue')) return 'Queued'
  if (s.includes('parse')) return 'Extracting text'
  if (s.includes('chunk')) return 'Building chunks'
  if (s.includes('embed')) return 'Embedding and indexing'
  if (s.includes('ready') || s.includes('complete') || s === 'done') return 'Ready'
  if (s.includes('fail') || s.includes('error')) return 'Failed'
  return 'In progress'
}

export function updateNoteStatusFromJob(job, { resourceId, jobId } = {}) {
  if (!job) return
  const id = resourceId || job.resource_id
  const jobKey = jobId || job.id
  const notes = loadNotesMetadata()
  if (!notes.length) return
  const label = mapJobStatusToLabel(job.status || job.state)
  let changed = false
  const nextNotes = notes.map((n) => {
    if (id && n.id === id) {
      changed = true
      return { ...n, status: label, last_job_status: job.status || job.state }
    }
    if (!id && jobKey && n.job_id === jobKey) {
      changed = true
      return { ...n, status: label, last_job_status: job.status || job.state }
    }
    return n
  })
  if (changed) saveNotesMetadata(nextNotes)
}
