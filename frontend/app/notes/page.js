"use client"

import { useEffect, useState } from 'react'
import { useAuth } from '../hooks/useAuth'
import { API_BASE } from '../lib/api'
import {
  loadNotesMetadata,
  saveNotesMetadata,
  upsertNoteMetadata,
  mapJobStatusToLabel,
} from '../lib/notes'

function authHeader(token) {
  return token ? `Bearer ${token}` : 'Bearer test-token'
}

function mapBackendNote(r) {
  const backendStatus = r.status || ''
  return {
    id: r.resource_id,
    name: r.original_filename || r.resource_id,
    status: mapJobStatusToLabel(backendStatus),
    backend_status: backendStatus,
    created_at: r.created_at || null,
    error_message: r.error_message || null,
  }
}

export default function NotesPage() {
  const { token } = useAuth({ requireAuth: true })
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [notes, setNotes] = useState([])
  const [error, setError] = useState(null)
  const [retryingId, setRetryingId] = useState(null)
  const [deletingId, setDeletingId] = useState(null)

  const MAX_BYTES = 100 * 1024 * 1024 // 100MB

  async function loadFromBackend() {
    if (!token) return
    try {
      const res = await fetch(`${API_BASE}/api/notes`, {
        headers: {
          Authorization: authHeader(token),
        },
      })
      if (!res.ok) throw new Error(`Notes list HTTP ${res.status}`)
      const data = await res.json()
      if (!Array.isArray(data)) return
      const mapped = data.map(mapBackendNote)
      setNotes(mapped)
      saveNotesMetadata(mapped)
    } catch (e) {
      // keep any locally cached notes but surface error
      setError(String(e))
    }
  }

  useEffect(() => {
    void loadFromBackend()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token])

  useEffect(() => {
    const cached = loadNotesMetadata()
    if (cached && cached.length) {
      setNotes(cached)
    }
  }, [])

  useEffect(() => {
    // Poll for notes that are still in progress
    const hasPending = notes.some((n) => {
      const s = String(n.backend_status || n.status || '').toLowerCase()
      if (!s) return false
      if (s.includes('ready') || s.includes('fail') || s.includes('error')) return false
      return true
    })
    if (!hasPending || !token) return

    const id = setInterval(() => {
      void loadFromBackend()
    }, 4000)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notes, token])

  function onFileChange(e) {
    setError(null)
    const f = e.target.files && e.target.files[0]
    if (!f) return
    if (f.size > MAX_BYTES) {
      setError('File exceeds 100MB limit')
      setFile(null)
      return
    }
    setFile(f)
  }

  async function handleUpload() {
    if (!file || !token) return
    setError(null)
    setUploading(true)

    try {
      const formData = new FormData()
      formData.append('file', file)

      // 1) Upload file directly to backend (backend stores to MinIO/GCS)
      const res = await fetch(`${API_BASE}/api/notes/upload`, {
        method: 'POST',
        headers: {
          Authorization: authHeader(token),
        },
        body: formData,
      })
      if (!res.ok) throw new Error(`upload HTTP ${res.status}`)
      const data = await res.json()

      const resourceId = data.resource_id
      const backendStatus = data.status || 'queued'
      const initialNote = {
        id: resourceId,
        name: file.name,
        status: mapJobStatusToLabel(backendStatus),
        backend_status: backendStatus,
        created_at: data.created_at || new Date().toISOString(),
        error_message: data.error_message || null,
      }
      setNotes((prev) => {
        const next = [initialNote, ...prev.filter((n) => n.id !== resourceId)]
        saveNotesMetadata(next)
        return next
      })
      upsertNoteMetadata(initialNote)

      // 2) Trigger ingestion for this resource
      const ingestRes = await fetch(`${API_BASE}/api/notes/${resourceId}/ingest`, {
        method: 'POST',
        headers: {
          Authorization: authHeader(token),
        },
      })
      if (!ingestRes.ok) {
        throw new Error(`ingest HTTP ${ingestRes.status}`)
      }

      // 3) Refresh list from backend; polling effect will keep it updated
      await loadFromBackend()
      setFile(null)
    } catch (e) {
      setError(String(e))
    } finally {
      setUploading(false)
    }
  }

  async function handleRetryIngest(noteId) {
    if (!noteId || !token) return
    setError(null)
    setRetryingId(noteId)
    try {
      setNotes((prev) => {
        const next = prev.map((n) => {
          if (n.id !== noteId) return n
          const backendStatus = 'parsing'
          return {
            ...n,
            backend_status: backendStatus,
            status: mapJobStatusToLabel(backendStatus),
            error_message: null,
          }
        })
        saveNotesMetadata(next)
        return next
      })

      const ingestRes = await fetch(`${API_BASE}/api/notes/${noteId}/ingest`, {
        method: 'POST',
        headers: {
          Authorization: authHeader(token),
        },
      })
      if (!ingestRes.ok) {
        throw new Error(`ingest HTTP ${ingestRes.status}`)
      }

      await loadFromBackend()
    } catch (e) {
      setError(String(e))
    } finally {
      setRetryingId(null)
    }
  }

  async function handleDeleteNote(noteId) {
    if (!noteId || !token) return
    setError(null)
    setDeletingId(noteId)
    try {
      const res = await fetch(`${API_BASE}/api/notes/${noteId}`, {
        method: 'DELETE',
        headers: {
          Authorization: authHeader(token),
        },
      })
      if (!res.ok) {
        throw new Error(`delete HTTP ${res.status}`)
      }

      setNotes((prev) => {
        const next = prev.filter((n) => n.id !== noteId)
        saveNotesMetadata(next)
        return next
      })
    } catch (e) {
      setError(String(e))
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <main style={{ padding: 24 }}>
      <h1>Your notes</h1>

      <section style={{ marginTop: 16, marginBottom: 24 }}>
        <h2>Upload new notes</h2>
        <input type="file" onChange={onFileChange} />
        {file && (
          <div>
            Selected: {file.name} ({Math.round(file.size / 1024)} KB)
          </div>
        )}
        <div style={{ marginTop: 8 }}>
          <button onClick={handleUpload} disabled={!file || uploading}>
            {uploading ? 'Uploading and ingesting…' : 'Upload and ingest'}
          </button>
        </div>
      </section>

      <section>
        <h2>Uploaded notes</h2>
        {notes.length === 0 && <div>No notes yet. Upload a PDF or notes file to get started.</div>}
        {notes.length > 0 && (
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {notes.map((n) => {
              const label = mapJobStatusToLabel(n.backend_status || n.status)
              return (
                <li
                  key={n.id}
                  style={{
                    border: '1px solid #ddd',
                    borderRadius: 4,
                    padding: 8,
                    marginBottom: 8,
                    background: '#fff',
                  }}
                >
                  <div style={{ fontWeight: 'bold' }}>{n.name}</div>
                  <div style={{ fontSize: 12, color: '#555' }}>
                    <span>Status: {label}</span>
                    {n.created_at && <span> • Uploaded: {new Date(n.created_at).toLocaleString()}</span>}
                  </div>
                  {n.error_message && (
                    <div style={{ fontSize: 12, color: 'red', marginTop: 4 }}>
                      Error: {n.error_message}
                    </div>
                  )}
                  <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {label === 'Failed' && (
                      <button
                        onClick={() => handleRetryIngest(n.id)}
                        disabled={uploading || retryingId === n.id}
                      >
                        {retryingId === n.id ? 'Retrying…' : 'Retry ingest'}
                      </button>
                    )}
                    <button
                      onClick={() => handleDeleteNote(n.id)}
                      disabled={deletingId === n.id}
                    >
                      {deletingId === n.id ? 'Deleting…' : 'Delete'}
                    </button>
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </section>

      {error && (
        <div style={{ color: 'red', marginTop: 16 }}>
          {error}
        </div>
      )}
    </main>
  )
}
