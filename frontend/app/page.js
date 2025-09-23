"use client"
import { useEffect, useState } from 'react'

export default function Home() {
  const [status, setStatus] = useState('loading')

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then((r) => r.json())
      .then((j) => setStatus(j.status))
      .catch(() => setStatus('error'))
  }, [])

  return (
    <main style={{ padding: 24 }}>
      <h1>StudyAgent Frontend</h1>
      <p>Backend status: {status}</p>
      <div style={{ marginTop: 16 }}>
        <strong>Quick Links</strong>
        <ul>
          <li><a href="/upload">Upload Resource</a></li>
          <li><a href="/agents/quiz">Quiz Agent</a></li>
          <li><a href="/agents/doubt">Doubt Chat</a></li>
          <li><a href="/agents/study-plan">Study Plan</a></li>
        </ul>
      </div>
    </main>
  )
}
