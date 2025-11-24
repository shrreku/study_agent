'use client';

import { useEffect, useRef, useState } from 'react';
import { useAuth } from '../../hooks/useAuth';
import { API_BASE } from '../../lib/api';

interface TutorMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: Date;
}

interface EnvironmentDebug {
  session?: {
    current_concept_index: number;
    concepts_completed: number;
    turn_count: number;
    terminated: boolean;
  };
  concept?: {
    concept_id: string;
    current_step_index: number;
    current_mastery: number;
    target_mastery: number;
    phase: string;
  };
  tutor?: {
    last_action: string | null;
    awaiting_user_input: boolean;
    turn_in_step: number;
  };
}

interface TutorResponse {
  messages: TutorMessage[];
  ui_mode: string;
  button_options: string[];
  debug?: EnvironmentDebug;
}

const AVAILABLE_MODELS = [
  { id: 'anthropic/claude-haiku-4.5', name: 'Claude Haiku 4.5 (Default)' },
  { id: 'gpt-4o-mini', name: 'GPT-4o Mini' },
  { id: 'gpt-4o', name: 'GPT-4o' },
];

export function EnvironmentTutorPage() {
  const { token } = useAuth({ requireAuth: true });
  
  const [sessionId, setSessionId] = useState('');
  const [targetConcepts, setTargetConcepts] = useState('Heat_Transfer,Convection,Conduction');
  const [selectedModel, setSelectedModel] = useState('anthropic/claude-haiku-4.5');
  const [messages, setMessages] = useState<TutorMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [debug, setDebug] = useState<EnvironmentDebug | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const authHeader = () => token ? `Bearer ${token}` : 'Bearer test-token';

  const startSession = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Create new session
      const res = await fetch(`${API_BASE}/api/tutor/session/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': authHeader(),
        },
        body: JSON.stringify({
          mode: 'general',
          agent_action_mode: 'step_by_step',
          target_concepts: targetConcepts.split(',').map(c => c.trim()).filter(Boolean),
        }),
      });

      if (!res.ok) {
        throw new Error(`Failed to start session: ${res.statusText}`);
      }

      const data = await res.json();
      setSessionId(data.session_id);
      
      // Send initial message to start learning
      await sendMessage('Start learning', data.session_id, true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start session');
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async (msg: string, sid?: string, isInitial = false) => {
    const currentSessionId = sid || sessionId;
    if (!currentSessionId) {
      setError('Please start a session first');
      return;
    }

    setLoading(true);
    setError(null);

    if (!isInitial) {
      // Add user message to UI
      setMessages(prev => [...prev, {
        role: 'user',
        content: msg,
        timestamp: new Date(),
      }]);
    }

    try {
      const res = await fetch(
        `${API_BASE}/api/tutor/pedagogy/session/${currentSessionId}/message`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': authHeader(),
          },
          body: JSON.stringify({
            message: msg,
            confirmed_action: null,
            model_hint: selectedModel,
          }),
        }
      );

      if (!res.ok) {
        throw new Error(`Failed to send message: ${res.statusText}`);
      }

      const data: TutorResponse = await res.json();
      
      // Add assistant messages
      if (data.messages && data.messages.length > 0) {
        const assistantMessages = data.messages.map(m => ({
          ...m,
          timestamp: new Date(),
        }));
        setMessages(prev => [...prev, ...assistantMessages]);
      }

      // Update debug info
      if (data.debug) {
        setDebug(data.debug);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
    } finally {
      setLoading(false);
    }
  };

  const clickContinue = async () => {
    if (!sessionId) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch(
        `${API_BASE}/api/tutor/pedagogy/session/${sessionId}/message`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': authHeader(),
          },
          body: JSON.stringify({
            message: '',
            confirmed_action: 'continue',
            model_hint: selectedModel,
          }),
        }
      );

      if (!res.ok) {
        throw new Error(`Failed to continue: ${res.statusText}`);
      }

      const data: TutorResponse = await res.json();
      
      // Add assistant messages
      if (data.messages && data.messages.length > 0) {
        const assistantMessages = data.messages.map(m => ({
          ...m,
          timestamp: new Date(),
        }));
        setMessages(prev => [...prev, ...assistantMessages]);
      }

      // Update debug info
      if (data.debug) {
        setDebug(data.debug);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to continue');
    } finally {
      setLoading(false);
    }
  };

  const clickReplan = async () => {
    if (!sessionId) return;

    setLoading(true);
    setError(null);

    // Add user action indicator
    setMessages(prev => [...prev, {
      role: 'user',
      content: '[Requested Re-plan]',
      timestamp: new Date(),
    }]);

    try {
      const res = await fetch(
        `${API_BASE}/api/tutor/pedagogy/session/${sessionId}/message`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': authHeader(),
          },
          body: JSON.stringify({
            message: '',
            step_control: { type: 'replan_concept' },
            model_hint: selectedModel,
          }),
        }
      );

      if (!res.ok) {
        throw new Error(`Failed to replan: ${res.statusText}`);
      }

      const data: TutorResponse = await res.json();
      
      // Add assistant messages
      if (data.messages && data.messages.length > 0) {
        const assistantMessages = data.messages.map(m => ({
          ...m,
          timestamp: new Date(),
        }));
        setMessages(prev => [...prev, ...assistantMessages]);
      }

      // Update debug info
      if (data.debug) {
        setDebug(data.debug);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to replan');
    } finally {
      setLoading(false);
    }
  };

  const getMasteryColor = (mastery: number) => {
    if (mastery >= 0.8) return 'text-green-600';
    if (mastery >= 0.5) return 'text-yellow-600';
    return 'text-orange-600';
  };

  const getMasteryBg = (mastery: number) => {
    if (mastery >= 0.8) return 'bg-green-100';
    if (mastery >= 0.5) return 'bg-yellow-100';
    return 'bg-orange-100';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="max-w-6xl mx-auto p-4 md:p-6">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6 border border-blue-100">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl text-2xl">
                ✨
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Environment Tutor</h1>
                <p className="text-sm text-gray-600">3-Layer MDP Architecture</p>
              </div>
            </div>
            
            {sessionId && debug && (
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2 px-3 py-2 bg-blue-50 rounded-lg">
                  <span className="text-lg">🎯</span>
                  <span className="font-medium text-blue-900">
                    {debug.session?.concepts_completed || 0} / {(debug.session?.current_concept_index || 0) + 1} concepts
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Session Setup */}
        {!sessionId && (
          <div className="bg-white rounded-2xl shadow-lg p-8 mb-6 border border-blue-100">
            <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
              <span className="text-xl">📚</span>
              Start Your Learning Session
            </h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Target Concepts (comma-separated)
                </label>
                <input
                  type="text"
                  value={targetConcepts}
                  onChange={(e) => setTargetConcepts(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="e.g., Heat_Transfer, Convection, Conduction"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  LLM Model
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                >
                  {AVAILABLE_MODELS.map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={startSession}
                disabled={loading || !targetConcepts.trim()}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-4 rounded-xl font-semibold hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white" />
                    Starting...
                  </>
                ) : (
                  <>
                    <span className="text-xl">▶</span>
                    Start Learning
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Progress Panel */}
        {sessionId && debug && debug.concept && (
          <div className="bg-white rounded-2xl shadow-lg p-6 mb-6 border border-blue-100">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Current Concept */}
              <div className="flex items-start gap-3">
                <div className="p-2 bg-blue-100 rounded-lg text-xl">
                  🧠
                </div>
                <div className="flex-1">
                  <p className="text-xs text-gray-600 font-medium uppercase tracking-wide mb-1">
                    Current Concept
                  </p>
                  <p className="text-sm font-bold text-gray-900">
                    {debug.concept.concept_id.replace(/_/g, ' ')}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Phase: {debug.concept.phase}
                  </p>
                </div>
              </div>

              {/* Mastery Progress */}
              <div className="flex items-start gap-3">
                <div className={`p-2 ${getMasteryBg(debug.concept.current_mastery)} rounded-lg text-xl`}>
                  {debug.concept.current_mastery >= 0.8 ? '✅' : debug.concept.current_mastery >= 0.5 ? '⚡' : '📈'}
                </div>
                <div className="flex-1">
                  <p className="text-xs text-gray-600 font-medium uppercase tracking-wide mb-1">
                    Mastery Level
                  </p>
                  <div className="flex items-baseline gap-2">
                    <span className={`text-sm font-bold ${getMasteryColor(debug.concept.current_mastery)}`}>
                      {(debug.concept.current_mastery * 100).toFixed(0)}%
                    </span>
                    <span className="text-xs text-gray-500">
                      / {(debug.concept.target_mastery * 100).toFixed(0)}% target
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                    <div
                      className={`h-2 rounded-full transition-all ${
                        debug.concept.current_mastery >= 0.8
                          ? 'bg-green-500'
                          : debug.concept.current_mastery >= 0.5
                          ? 'bg-yellow-500'
                          : 'bg-orange-500'
                      }`}
                      style={{ width: `${debug.concept.current_mastery * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Step Progress */}
              <div className="flex items-start gap-3">
                <div className="p-2 bg-purple-100 rounded-lg text-xl">
                  💬
                </div>
                <div className="flex-1">
                  <p className="text-xs text-gray-600 font-medium uppercase tracking-wide mb-1">
                    Current Step
                  </p>
                  <p className="text-sm font-bold text-gray-900">
                    Step {debug.concept.current_step_index + 1}
                  </p>
                  {debug.tutor?.last_action && (
                    <p className="text-xs text-gray-500 mt-1">
                      Action: {debug.tutor.last_action.toLowerCase()}
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-xl mb-6">
            <p className="text-sm font-medium">{error}</p>
          </div>
        )}

        {/* Messages */}
        {sessionId && (
          <div className="bg-white rounded-2xl shadow-lg border border-blue-100 overflow-hidden">
            <div className="h-[500px] overflow-y-auto p-6 space-y-4">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-5 py-3 ${
                      msg.role === 'user'
                        ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white'
                        : 'bg-gray-100 text-gray-900'
                    }`}
                  >
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                  </div>
                </div>
              ))}
              
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 rounded-2xl px-5 py-3">
                    <div className="flex items-center gap-2">
                      <div className="animate-bounce">●</div>
                      <div className="animate-bounce delay-100">●</div>
                      <div className="animate-bounce delay-200">●</div>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>

            {/* Control Buttons */}
            <div className="border-t border-gray-200 p-4 bg-gray-50">
              <div className="flex gap-3">
                <button
                  onClick={clickContinue}
                  disabled={loading}
                  className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-4 rounded-xl font-semibold hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white" />
                      Processing...
                    </>
                  ) : (
                    <>
                      Continue
                      <span className="text-xl">→</span>
                    </>
                  )}
                </button>
                
                <button
                  onClick={clickReplan}
                  disabled={loading}
                  className="bg-white text-gray-700 border border-gray-300 px-6 py-4 rounded-xl font-semibold hover:bg-gray-50 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md flex items-center justify-center gap-2"
                >
                  <span className="text-xl">🔄</span>
                  Re-plan
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
