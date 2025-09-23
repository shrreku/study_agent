"use client"
import { useState, useRef, useEffect } from 'react'

export default function DoubtChatPage() {
  const [question, setQuestion] = useState('Explain the lumped capacitance method')
  const [messages, setMessages] = useState([]) // {role:'user'|'assistant', content, citations?}
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const endRef = useRef(null)

  useEffect(() => { if (endRef.current) endRef.current.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  async function ask() {
    if (!question.trim()) return
    setError(null)
    const q = question.trim()
    setQuestion('')
    setMessages(prev => [...prev, { role: 'user', content: q }])
    try {
      setLoading(true)
      const res = await fetch('http://localhost:8000/api/agent/doubt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer test-token' },
        body: JSON.stringify({ question: q }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer || '', citations: data.citations || [] }])
    } catch (e) {
      setError(String(e))
      setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, couldn't answer right now.", citations: [] }])
    } finally {
      setLoading(false)
    }
  }

  function onKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      ask()
    }
  }

  return (
    <main style={{ padding: 24 }}>
      <h1>Doubt Chat</h1>
      <div style={{ maxWidth: 800, margin: '12px 0' }}>
        <textarea value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={onKey} rows={3} style={{ width: '100%' }} placeholder="Ask a question..." />
        <div style={{ marginTop: 8 }}>
          <button onClick={ask} disabled={loading || !question.trim()}>Ask</button>
        </div>
      </div>
      {error && <div style={{ color: 'red' }}>{error}</div>}

      <div style={{ maxWidth: 900, background: '#fafafa', border: '1px solid #eee', padding: 12, minHeight: 200 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 12 }}>
            <div style={{ color: m.role === 'user' ? '#333' : '#0b6' }}><strong>{m.role === 'user' ? 'You' : 'Tutor'}</strong></div>
            <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
            {m.role === 'assistant' && Array.isArray(m.citations) && m.citations.length > 0 && (
              <div style={{ marginTop: 6, fontSize: 12, color: '#555' }}>
                <strong>Citations</strong>
                <ul>
                  {m.citations.map((c, idx) => (
                    <li key={idx}><code>{c.chunk_id}</code> — {c.page ? `p.${c.page} — ` : ''}{(c.snippet || '').slice(0, 160)}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        <div ref={endRef} />
        {loading && <div>Thinking…</div>}
      </div>
    </main>
  )
}
