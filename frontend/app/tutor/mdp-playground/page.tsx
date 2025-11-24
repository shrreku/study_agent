'use client';

import { useState } from 'react';
import { useAuth } from '../../hooks/useAuth';
import { API_BASE } from '../../lib/api';

const AVAILABLE_MODELS = [
  { id: 'anthropic/claude-haiku-4.5', name: 'Claude Haiku 4.5 (Default)' },
  { id: 'gpt-4o-mini', name: 'GPT-4o Mini' },
  { id: 'gpt-4o', name: 'GPT-4o' },
];

const TUTOR_ACTIONS = [
  'EXPLAIN',
  'ASK_QUESTION',
  'WORKED_EXAMPLE',
  'GUIDED_PRACTICE',
  'DEFINE_TERM',
  'REFLECTION_PROMPT',
  'SUMMARY',
  'QUIZ_MCQ',
];

interface TutorMessage {
  role: 'assistant';
  content: string;
}

interface PlaygroundResponse {
  messages?: TutorMessage[];
  ui_mode?: string;
  mcq_payload?: any;
  debug?: Record<string, any>;
}

export default function TutorMDPPlaygroundPage() {
  const { token } = useAuth({ requireAuth: true });

  const [conceptId, setConceptId] = useState('Heat_Transfer');
  const [selectedModel, setSelectedModel] = useState('anthropic/claude-haiku-4.5');
  const [planActions, setPlanActions] = useState<string[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [messages, setMessages] = useState<TutorMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const authHeader = () => (token ? `Bearer ${token}` : 'Bearer test-token');

  const generateRandomPlan = () => {
    const length = 4;
    const actions: string[] = [];
    for (let i = 0; i < length; i += 1) {
      const idx = Math.floor(Math.random() * TUTOR_ACTIONS.length);
      actions.push(TUTOR_ACTIONS[idx]);
    }
    setPlanActions(actions);
    setCurrentIndex(0);
    setMessages([]);
  };

  const hasPlan = planActions.length > 0;
  const currentAction = hasPlan && currentIndex < planActions.length ? planActions[currentIndex] : null;

  const stepOnce = async () => {
    if (!hasPlan || currentAction == null) {
      setError('Generate a plan first.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/api/tutor/mdp/playground/step`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: authHeader(),
        },
        body: JSON.stringify({
          concept_id: conceptId,
          pedagogical_action: currentAction,
          plan_index: currentIndex,
          plan_length: planActions.length,
          phase: 'learning',
          student_message: '',
          model_hint: selectedModel,
        }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Request failed: ${res.status} ${text}`);
      }

      const data: PlaygroundResponse = await res.json();

      if (data.messages && data.messages.length > 0) {
        setMessages((prev) => [...prev, ...data.messages]);
      }

      if (currentIndex < planActions.length - 1) {
        setCurrentIndex((idx) => idx + 1);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to execute step');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="max-w-5xl mx-auto px-4 py-6 space-y-6">
        <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Tutor MDP Playground</p>
            <h1 className="text-2xl font-bold text-slate-900">Pedagogical Action Runner</h1>
            <p className="text-sm text-slate-600">
              Select an LLM, generate a random sequence of tutor actions, and step through them.
            </p>
          </div>
        </header>

        <section className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Concept ID</label>
              <input
                type="text"
                value={conceptId}
                onChange={(e) => setConceptId(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">LLM Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {AVAILABLE_MODELS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-end gap-2">
              <button
                type="button"
                onClick={generateRandomPlan}
                className="w-full bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-semibold hover:bg-blue-700 transition disabled:opacity-50"
                disabled={loading}
              >
                Generate Random Plan
              </button>
            </div>
          </div>

          {error && (
            <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2">{error}</div>
          )}
        </section>

        {hasPlan && (
          <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4 space-y-2 md:col-span-1">
              <h2 className="text-sm font-semibold text-slate-800 mb-2">Plan Actions</h2>
              <ol className="space-y-1 text-sm">
                {planActions.map((action, idx) => {
                  const isCurrent = idx === currentIndex;
                  return (
                    <li
                      key={`${action}-${idx}`}
                      className={`flex items-center justify-between px-3 py-2 rounded-lg border text-xs font-mono ${
                        isCurrent
                          ? 'border-blue-500 bg-blue-50 text-blue-900'
                          : 'border-slate-200 bg-slate-50 text-slate-800'
                      }`}
                    >
                      <span>
                        {idx + 1}. {action}
                      </span>
                      {isCurrent && <span className="text-[10px] font-semibold uppercase">Current</span>}
                    </li>
                  );
                })}
              </ol>

              <button
                type="button"
                onClick={stepOnce}
                disabled={loading || !currentAction}
                className="mt-4 w-full bg-purple-600 text-white px-4 py-2 rounded-lg text-sm font-semibold hover:bg-purple-700 transition disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {loading ? 'Running...' : 'Continue (Next Action)'}
              </button>
            </div>

            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4 space-y-3 md:col-span-2">
              <h2 className="text-sm font-semibold text-slate-800 mb-2">Tutor Responses</h2>
              <div className="h-64 overflow-y-auto space-y-3 text-sm">
                {messages.length === 0 && (
                  <p className="text-xs text-slate-500">No responses yet. Click Continue to run the first action.</p>
                )}
                {messages.map((m, idx) => (
                  <div
                    key={idx}
                    className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 whitespace-pre-wrap"
                  >
                    {m.content}
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
