"use client"

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { useAuth } from '../hooks/useAuth'
import { API_BASE } from '../lib/api'
import { loadNotesMetadata, saveNotesMetadata, mapJobStatusToLabel, updateNoteStatusFromJob } from '../lib/notes'
import { Card, PrimaryButton, SecondaryButton } from '../components/ui'

export default function DashboardPage() {
  const { user, token, loading } = useAuth({ requireAuth: true })
  const [notes, setNotes] = useState([])
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)

  function authHeader() {
    return token ? `Bearer ${token}` : 'Bearer test-token'
  }

  useEffect(() => {
    if (typeof window === 'undefined') return
    const stored = loadNotesMetadata()
    setNotes(stored)
  }, [])

  async function refreshStatuses() {
    if (!notes.length) return
    setRefreshing(true)
    setError(null)
    try {
      const updated = [...notes]
      for (let i = 0; i < updated.length; i += 1) {
        const n = updated[i]
        if (!n.job_id) continue
        try {
          const res = await fetch(`${API_BASE}/api/jobs/${n.job_id}`, {
            headers: { Authorization: authHeader() },
          })
          if (!res.ok) continue
          const job = await res.json()
          updateNoteStatusFromJob(job, { resourceId: n.id, jobId: n.job_id })
          const label = mapJobStatusToLabel(job.status || job.state)
          updated[i] = { ...n, status: label, last_job_status: job.status || job.state }
        } catch (e) {
          // ignore individual failures
        }
      }
      setNotes(updated)
      saveNotesMetadata(updated)
    } catch (e) {
      setError(String(e))
    } finally {
      setRefreshing(false)
    }
  }

  const totalNotes = notes.length
  const readyNotes = useMemo(() => notes.filter((n) => n.status === 'Ready').length, [notes])

  return (
    <main>
      <h1 style={{ fontSize: 28, marginBottom: 8 }}>Dashboard</h1>
      <p style={{ marginTop: 0, marginBottom: 16, color: '#4b5563', maxWidth: 640 }}>
        Welcome{user?.email ? `, ${user.email}` : ''}. Upload notes, track ingestion, and chat with the tutor.
      </p>

      {error && (
        <div style={{ marginBottom: 12, fontSize: 13, color: '#b91c1c' }}>{error}</div>
      )}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
        <Card
          title="My Notes"
          subtitle="Uploads and ingestion status"
          style={{ flex: '1 1 280px', minWidth: 260 }}
          footer={
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <PrimaryButton onClick={refreshStatuses} disabled={refreshing || !notes.length}>
                {refreshing ? 'Refreshing…' : 'Refresh status'}
              </PrimaryButton>
              <SecondaryButton as="a" href="/notes">
                Go to notes
              </SecondaryButton>
            </div>
          }
        >
          <div style={{ marginBottom: 8, fontSize: 14 }}>
            <div>Total uploads: <strong>{totalNotes}</strong></div>
            <div>Ready: <strong>{readyNotes}</strong></div>
          </div>
          {notes.length === 0 ? (
            <p style={{ fontSize: 13, color: '#6b7280', marginBottom: 0 }}>
              No notes yet. Start by uploading a PDF or slide deck on the <Link href="/notes">Notes</Link> page.
            </p>
          ) : (
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              {notes.slice(0, 4).map((n) => (
                <li
                  key={n.id}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '6px 0',
                    borderTop: '1px solid #f3f4f6',
                    fontSize: 13,
                  }}
                >
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {n.name || n.id}
                    </div>
                    <div style={{ fontSize: 11, color: '#9ca3af' }}>{n.status || 'Uploaded'}</div>
                  </div>
                  {n.status && (
                    <span
                      style={{
                        marginLeft: 8,
                        padding: '2px 8px',
                        borderRadius: 999,
                        border: '1px solid #e5e7eb',
                        fontSize: 11,
                        color:
                          n.status === 'Ready'
                            ? '#166534'
                            : n.status.startsWith('Failed')
                            ? '#b91c1c'
                            : '#4b5563',
                        background:
                          n.status === 'Ready'
                            ? '#ecfdf3'
                            : n.status.startsWith('Failed')
                            ? '#fef2f2'
                            : '#f3f4f6',
                      }}
                    >
                      {n.status}
                    </span>
                  )}
                </li>
              ))}
              {notes.length > 4 && (
                <li style={{ marginTop: 4, fontSize: 12, color: '#6b7280' }}>
                  + {notes.length - 4} more
                </li>
              )}
            </ul>
          )}
        </Card>

        <Card
          title="Start Tutor"
          subtitle="Study with or without your notes"
          style={{ flex: '1 1 260px', minWidth: 260 }}
        >
          <p style={{ fontSize: 13, color: '#4b5563', marginTop: 0 }}>
            Use the tutor to revise concepts, generate explanations, and ask questions. You can ground answers in
            your uploaded notes or use a general tutor mode.
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
            <PrimaryButton onClick={() => (window.location.href = '/tutor')}>
              Study my notes
            </PrimaryButton>
            <SecondaryButton onClick={() => (window.location.href = '/tutor')}>
              General tutor
            </SecondaryButton>
          </div>
        </Card>
      </div>
    </main>
  )
}
