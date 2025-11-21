"use client"

import Link from 'next/link'
import { useAuth } from '../hooks/useAuth'

export function NavBar() {
  const { user, loading, logout } = useAuth()

  const userLabel = user?.email || user?.display_name || user?.id || null

  return (
    <header
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 20,
        borderBottom: '1px solid #e5e7eb',
        background: 'rgba(249, 250, 251, 0.9)',
        backdropFilter: 'blur(12px)',
      }}
    >
      <nav
        style={{
          maxWidth: 960,
          margin: '0 auto',
          padding: '12px 16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 16,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <Link href="/dashboard" style={{ textDecoration: 'none', color: '#111827' }}>
            <span style={{ fontWeight: 600, letterSpacing: '-0.02em' }}>StudyAgent</span>
          </Link>
          <span style={{ fontSize: 12, padding: '2px 8px', borderRadius: 999, background: '#e5e7eb', color: '#4b5563' }}>
            beta
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 14 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            <Link href="/dashboard" style={{ color: '#374151', textDecoration: 'none' }}>
              Dashboard
            </Link>
            <Link href="/notes" style={{ color: '#374151', textDecoration: 'none' }}>
              Notes
            </Link>
            <Link href="/tutor" style={{ color: '#374151', textDecoration: 'none' }}>
              Tutor
            </Link>
          </div>
          <div style={{ flex: 1 }} />
          {loading ? (
            <span style={{ fontSize: 13, color: '#6b7280' }}>Loading…</span>
          ) : userLabel ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 13, color: '#4b5563' }}>{userLabel}</span>
              <button
                type="button"
                onClick={logout}
                style={{
                  fontSize: 12,
                  padding: '4px 10px',
                  borderRadius: 999,
                  border: '1px solid #e5e7eb',
                  background: '#f9fafb',
                  cursor: 'pointer',
                }}
              >
                Logout
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Link href="/login" style={{ fontSize: 13, color: '#4b5563', textDecoration: 'none' }}>
                Login
              </Link>
              <span style={{ fontSize: 12, color: '#d1d5db' }}>/</span>
              <Link href="/register" style={{ fontSize: 13, color: '#111827', textDecoration: 'none' }}>
                Sign up
              </Link>
            </div>
          )}
        </div>
      </nav>
    </header>
  )
}
