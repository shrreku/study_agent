'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useAuth } from '../../hooks/useAuth';
import { API_BASE } from '../../lib/api';
import { TutorStateHeader } from './TutorStateHeader';
import { TutorMessage } from './TutorMessage';
import { useTutorSettings } from '../../hooks/useTutorSettings';
import {
  TutorResponse,
  TutorState,
  ConversationTurn,
  MasteryInfo,
} from '../../types/tutor';

function newId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function parseConcepts(input: string): string[] {
  return (input || '')
    .split(',')
    .map((c) => c.trim())
    .filter(Boolean);
}

/**
 * Enhanced tutor testing page with citation UI and state visualization
 */
export function TutorTestingPage() {
  const { token } = useAuth({ requireAuth: true });
  const { settings, isLoaded, updateSettings } = useTutorSettings();

  const [userId, setUserId] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [resourceId, setResourceId] = useState('');
  const [targetConceptsInput, setTargetConceptsInput] = useState('');
  const [message, setMessage] = useState('What concept should I revise today?');
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentState, setCurrentState] = useState<TutorState>('orientation');
  const [masteryMap, setMasteryMap] = useState<MasteryInfo>({});

  const endRef = useRef<HTMLDivElement>(null);

  // Load cached data
  useEffect(() => {
    try {
      const cachedUser = window.localStorage.getItem('tutor_user_id');
      const cachedSession = window.localStorage.getItem('tutor_session_id');
      const cachedResource = window.localStorage.getItem('tutor_resource_id');
      const cachedConcepts = window.localStorage.getItem('tutor_target_concepts');
      if (cachedUser) setUserId(cachedUser);
      if (cachedSession) setSessionId(cachedSession);
      if (cachedResource) setResourceId(cachedResource);
      if (cachedConcepts) setTargetConceptsInput(cachedConcepts);
    } catch (e) {
      console.warn('Failed to load cached tutor data', e);
    }
  }, []);

  // Auto-scroll to latest message
  useEffect(() => {
    if (endRef.current) {
      endRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [turns]);

  function authHeader(): string {
    return token ? `Bearer ${token}` : 'Bearer test-token';
  }

  function persistLocal(key: string, value: string | null) {
    try {
      if (value) {
        window.localStorage.setItem(key, value);
      } else {
        window.localStorage.removeItem(key);
      }
    } catch (e) {
      console.warn(`Failed to persist ${key}`, e);
    }
  }

  async function sendMessage() {
    if (loading) return;
    const trimmedMessage = message.trim();
    const trimmedUser = userId.trim();
    if (!trimmedMessage || !trimmedUser) {
      setError('Provide both a message and user ID before sending.');
      return;
    }

    const payload: Record<string, any> = {
      message: trimmedMessage,
      user_id: trimmedUser,
    };

    const trimmedSession = sessionId.trim();
    const trimmedResource = resourceId.trim();
    const targetConcepts = parseConcepts(targetConceptsInput);

    if (trimmedSession) payload.session_id = trimmedSession;
    if (trimmedResource) payload.resource_id = trimmedResource;
    if (targetConcepts.length > 0) payload.target_concepts = targetConcepts;

    setError(null);
    setMessage('');

    // Add user message to conversation
    setTurns((prev) => [
      ...prev,
      {
        id: newId(),
        role: 'user',
        content: trimmedMessage,
      },
    ]);

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/agent/tutor`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: authHeader(),
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        let detail = '';
        try {
          const errorPayload = await res.json();
          if (errorPayload?.detail) {
            detail = Array.isArray(errorPayload.detail)
              ? errorPayload.detail.join(', ')
              : String(errorPayload.detail);
          }
        } catch (parseErr) {
          try {
            const text = await res.text();
            detail = text?.slice(0, 400) || '';
          } catch (e) {
            /* ignore */
          }
        }
        const errMsg = detail
          ? `Tutor agent error (${res.status}): ${detail}`
          : `Tutor agent HTTP ${res.status}`;
        throw new Error(errMsg);
      }

      const data = await res.json();

      // Parse tutor response with new citation fields
      const tutorResponse: TutorResponse = {
        response_text: data.response || '',
        action: data.action || data.action_type || 'explain',
        confidence: data.confidence || 0.5,
        source_chunk_ids: data.source_chunk_ids || [],
        action_type: data.action_type,
        concept: data.concept,
        level: data.level,
        learning_path: data.learning_path,
        cold_start: data.cold_start,
        intent: data.intent,
        affect: data.affect,
        classification_confidence: data.classification_confidence,

        // New fields from enhanced backend
        citations: data.citations,
        grounding_mode: data.grounding_mode,
        decision_context: data.decision_context,
      };

      // Update state if provided
      if (data.current_state) {
        setCurrentState(data.current_state as TutorState);
      }

      // Update mastery if provided
      if (data.mastery_map) {
        setMasteryMap(data.mastery_map as MasteryInfo);
      }

      const tutorTurn: ConversationTurn = {
        id: newId(),
        role: 'tutor',
        content: tutorResponse.response_text,
        meta: {
          actionType: tutorResponse.action_type,
          confidence: tutorResponse.confidence,
          concept: tutorResponse.concept,
          level: tutorResponse.level,
          learningPath: tutorResponse.learning_path,
          coldStart: tutorResponse.cold_start,
          intent: tutorResponse.intent,
          affect: tutorResponse.affect,
          classificationConfidence: tutorResponse.classification_confidence,
          sourceChunks: tutorResponse.source_chunk_ids,
          citations: tutorResponse.citations,
          decisionContext: tutorResponse.decision_context,
          state: data.current_state,
          raw: data,
        },
      };

      setTurns((prev) => [...prev, tutorTurn]);

      if (data.session_id) {
        setSessionId(data.session_id);
        persistLocal('tutor_session_id', data.session_id);
      }
      persistLocal('tutor_user_id', trimmedUser);
      persistLocal('tutor_resource_id', trimmedResource);
      persistLocal('tutor_target_concepts', targetConceptsInput);
    } catch (err) {
      console.error('tutor_agent_call_failed', err);
      setError(String(err));
      setTurns((prev) => [
        ...prev,
        {
          id: newId(),
          role: 'tutor',
          content: "Sorry, I couldn't respond right now.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyPress(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!loading) sendMessage();
    }
  }

  function resetSession() {
    setSessionId('');
    setTurns([]);
    persistLocal('tutor_session_id', null);
  }

  function clearAll() {
    setSessionId('');
    setUserId('');
    setResourceId('');
    setTargetConceptsInput('');
    setTurns([]);
    setCurrentState('orientation');
    setMasteryMap({});
    try {
      window.localStorage.removeItem('tutor_session_id');
      window.localStorage.removeItem('tutor_user_id');
      window.localStorage.removeItem('tutor_resource_id');
      window.localStorage.removeItem('tutor_target_concepts');
    } catch (e) {
      console.warn('Failed to clear localStorage', e);
    }
  }

  function downloadTranscript() {
    const transcript = {
      generated_at: new Date().toISOString(),
      user_id: userId,
      session_id: sessionId,
      resource_id: resourceId,
      target_concepts: parseConcepts(targetConceptsInput),
      final_state: currentState,
      final_mastery: masteryMap,
      turns: turns.map((t) => ({
        role: t.role,
        content: t.content,
        meta: t.meta,
      })),
    };
    const blob = new Blob([JSON.stringify(transcript, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `tutor-session-${sessionId || 'new'}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  const stateHistory = useMemo(
    () => turns.filter((t) => t.meta?.state).map((t) => t.meta!.state as TutorState),
    [turns],
  );

  if (!isLoaded) {
    return <div className="p-6">Loading settings...</div>;
  }

  return (
    <main className="max-w-6xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="mb-6">
        <h1 className="text-4xl font-bold mb-2">🎓 Enhanced Tutor Dashboard</h1>
        <p className="text-gray-600">
          With state visualization, citations, and decision context
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Controls */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-md p-6 space-y-4">
            <h2 className="text-xl font-bold mb-4">⚙️ Session Setup</h2>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                User ID *
              </label>
              <input
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="required"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Session ID
              </label>
              <input
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                placeholder="autofills after first response"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Resource ID
              </label>
              <input
                value={resourceId}
                onChange={(e) => setResourceId(e.target.value)}
                placeholder="optional"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Target Concepts
              </label>
              <input
                value={targetConceptsInput}
                onChange={(e) => setTargetConceptsInput(e.target.value)}
                placeholder="e.g. conduction, convection"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="space-y-2 border-t pt-4">
              <h3 className="font-medium text-gray-700">Display Options</h3>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={settings.showCitations}
                  onChange={(e) => updateSettings({ showCitations: e.target.checked })}
                  className="rounded"
                />
                <span className="text-sm text-gray-600">Show Citations</span>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={settings.showDecisionContext}
                  onChange={(e) =>
                    updateSettings({ showDecisionContext: e.target.checked })
                  }
                  className="rounded"
                />
                <span className="text-sm text-gray-600">Show Decision Context</span>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={settings.debugMode}
                  onChange={(e) => updateSettings({ debugMode: e.target.checked })}
                  className="rounded"
                />
                <span className="text-sm text-gray-600">Debug Mode</span>
              </label>
            </div>

            <div className="flex flex-col gap-2">
              <button
                onClick={sendMessage}
                disabled={loading || !message.trim() || !userId.trim()}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 rounded-lg transition"
              >
                {loading ? 'Sending...' : 'Send Message'}
              </button>
              <button
                onClick={resetSession}
                disabled={loading}
                className="w-full bg-gray-500 hover:bg-gray-600 disabled:bg-gray-400 text-white font-medium py-2 rounded-lg transition"
              >
                Reset Session
              </button>
              <button
                onClick={clearAll}
                disabled={loading}
                className="w-full bg-red-500 hover:bg-red-600 disabled:bg-gray-400 text-white font-medium py-2 rounded-lg transition"
              >
                Clear All
              </button>
              <button
                onClick={downloadTranscript}
                disabled={turns.length === 0}
                className="w-full bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white font-medium py-2 rounded-lg transition"
              >
                Download
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Chat and State */}
        <div className="lg:col-span-2 space-y-6">
          {/* State Header */}
          {settings.showStateHeader && (
            <TutorStateHeader
              currentState={currentState}
              masteryMap={masteryMap}
              mode="intelligent"
              stateHistory={stateHistory}
            />
          )}

          {/* Message Input */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              rows={3}
              placeholder="Ask or respond to the tutor..."
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {error && <div className="text-red-600 text-sm mt-2">{error}</div>}
          </div>

          {/* Conversation */}
          <div className="bg-white rounded-lg shadow-md p-6 min-h-96 max-h-96 overflow-y-auto">
            {turns.length === 0 && (
              <div className="text-gray-500 text-center py-8">
                No messages yet. Start the session above.
              </div>
            )}

            {turns.map((turn) => (
              <div
                key={turn.id}
                className={`mb-4 pb-4 border-b border-gray-200 last:border-b-0`}
              >
                <div
                  className={`font-semibold mb-2 ${
                    turn.role === 'user' ? 'text-gray-900' : 'text-green-700'
                  }`}
                >
                  {turn.role === 'user' ? '👤 You' : '🤖 Tutor'}
                </div>

                {turn.role === 'tutor' && turn.meta ? (
                  <TutorMessage
                    message={{
                      response_text: turn.content,
                      action: turn.meta.actionType || 'explain',
                      confidence: turn.meta.confidence || 0.5,
                      citations: turn.meta.citations,
                      grounding_mode: 'explicit_citation',
                      decision_context: turn.meta.decisionContext,
                      source_chunk_ids: turn.meta.sourceChunks,
                    }}
                    showCitations={settings.showCitations}
                    showDecisionContext={settings.showDecisionContext}
                    debugMode={settings.debugMode}
                    onViewFullCitation={(chunkId) => {
                      // console.log('View full citation:', chunkId);
                    }}
                  />
                ) : (
                  <div className="text-gray-700 whitespace-pre-wrap">{turn.content}</div>
                )}
              </div>
            ))}

            {loading && <div className="text-gray-500 animate-pulse">Tutor is thinking...</div>}
            <div ref={endRef} />
          </div>
        </div>
      </div>
    </main>
  );
}

