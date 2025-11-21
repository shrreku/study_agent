'use client';

import React, { useState } from 'react';
import { TutorStateHeader } from './TutorStateHeader';
import { TutorMessage } from './TutorMessage';
import {
  TutorState,
  TutorResponse,
  Citation,
  MasteryInfo,
  DecisionContext,
} from '../../types/tutor';

/**
 * Demo component showing all tutor UI components in action
 * Used for development and testing
 */
export function TutorDemo() {
  const [currentState, setCurrentState] = useState<TutorState>('teaching');
  const [showCitations, setShowCitations] = useState(true);
  const [showDecisionContext, setShowDecisionContext] = useState(true);
  const [debugMode, setDebugMode] = useState(false);

  // Sample mastery map
  const masteryMap: MasteryInfo = {
    'Heat Transfer': 0.75,
    'Conduction': 0.62,
    'Convection': 0.85,
    'Radiation': 0.45,
  };

  // Sample citations
  const citations: Citation[] = [
    {
      chunk_id: 'chunk_001',
      snippet_text:
        'Heat transfer is the movement of thermal energy from one object or system to another. There are three primary modes of heat transfer: conduction, convection, and radiation. Each mode operates under different conditions and is governed by different physical principles.',
      relevance_score: 0.95,
      pedagogy_role: 'definition',
      page_number: 12,
      source_document: 'Heat Transfer Fundamentals.pdf',
    },
    {
      chunk_id: 'chunk_002',
      snippet_text:
        'A practical example of conduction can be observed when you hold a metal rod in a flame. The end of the rod that is not in the flame will eventually become hot as heat is conducted through the material. In contrast, convection involves the movement of fluid (liquid or gas) itself, as seen when water is heated in a pot.',
      relevance_score: 0.88,
      pedagogy_role: 'example',
      page_number: 15,
      source_document: 'Heat Transfer Fundamentals.pdf',
    },
  ];

  // Sample decision context
  const decisionContext: DecisionContext = {
    rationale:
      'You have received three consecutive explanations about conduction. To check if you understand the concept, I am now asking a question to assess your comprehension.',
    cause: '3 consecutive explains',
    trigger_type: 'state_transition',
  };

  // Sample tutor response
  const tutorResponse: TutorResponse = {
    response_text: `Great question about heat transfer! Based on what we've learned, can you explain the key difference between conduction and convection? Think about how heat moves in each case.`,
    action: 'ask',
    confidence: 0.92,
    citations,
    grounding_mode: 'explicit_citation',
    decision_context: decisionContext,
    source_chunk_ids: ['chunk_001', 'chunk_002'],
    action_type: 'assess_understanding',
    concept: 'Heat Transfer',
    level: 'intermediate',
    learning_path: ['Heat Transfer', 'Conduction', 'Convection'],
  };

  const stateHistory: TutorState[] = ['orientation', 'teaching', 'teaching', currentState];

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h1 className="text-3xl font-bold mb-2">Tutor UI Components Demo</h1>
        <p className="text-gray-600">
          Test and preview the new citation UI, state headers, and decision context
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg shadow p-6 space-y-4">
        <h2 className="text-xl font-bold mb-4">🎛️ Settings</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Current State
            </label>
            <select
              value={currentState}
              onChange={(e) => setCurrentState(e.target.value as TutorState)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="orientation">🧭 Orientation</option>
              <option value="teaching">🎓 Teaching</option>
              <option value="assessment">📝 Assessment</option>
              <option value="review">🔄 Review</option>
              <option value="closure">✅ Closure</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showCitations}
                onChange={(e) => setShowCitations(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm font-medium text-gray-700">Show Citations</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showDecisionContext}
                onChange={(e) => setShowDecisionContext(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm font-medium text-gray-700">Show Decision Context</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={debugMode}
                onChange={(e) => setDebugMode(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm font-medium text-gray-700">Debug Mode</span>
            </label>
          </div>
        </div>
      </div>

      {/* State Header */}
      <div className="space-y-3">
        <h2 className="text-xl font-bold">📊 State Header</h2>
        <TutorStateHeader
          currentState={currentState}
          masteryMap={masteryMap}
          mode="intelligent"
          stateHistory={stateHistory}
        />
      </div>

      {/* Tutor Message with All Features */}
      <div className="space-y-3">
        <h2 className="text-xl font-bold">💬 Tutor Response with Citations</h2>
        <TutorMessage
          message={tutorResponse}
          showCitations={showCitations}
          showDecisionContext={showDecisionContext}
          debugMode={debugMode}
          onViewFullCitation={(chunkId) => {
            // console.log('View full citation:', chunkId);
            alert(`Would navigate to view full context for chunk: ${chunkId}`);
          }}
        />
      </div>

      {/* Feature Explanation */}
      <div className="bg-white rounded-lg shadow p-6 space-y-4">
        <h2 className="text-xl font-bold mb-4">✨ Features Demonstrated</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="border-l-4 border-blue-500 pl-4">
            <h3 className="font-semibold text-gray-900 mb-2">State Header</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Shows current learning state (TEACHING, ASSESSMENT, etc.)</li>
              <li>• Displays mastery progress bars for concepts</li>
              <li>• Shows mode (simple, intelligent, step_by_step)</li>
              <li>• Optional state history path</li>
            </ul>
          </div>

          <div className="border-l-4 border-green-500 pl-4">
            <h3 className="font-semibold text-gray-900 mb-2">Citations</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Structured, cleaned snippets (no OCR artifacts)</li>
              <li>• Collapsible for better UX</li>
              <li>• Shows relevance scores</li>
              <li>• Pedagogy tags (definition, example, etc.)</li>
            </ul>
          </div>

          <div className="border-l-4 border-yellow-500 pl-4">
            <h3 className="font-semibold text-gray-900 mb-2">Decision Context</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Explains why this action was chosen</li>
              <li>• Shows trigger/cause (e.g., &quot;3 consecutive explains&quot;)</li>
              <li>• Indicates decision type (policy, heuristic, etc.)</li>
              <li>• Optional display for clean UI</li>
            </ul>
          </div>

          <div className="border-l-4 border-purple-500 pl-4">
            <h3 className="font-semibold text-gray-900 mb-2">Mastery Bars</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Visual progress for each concept (0-100%)</li>
              <li>• Color-coded by achievement level</li>
              <li>• Sorted by level (highest first)</li>
              <li>• Limited to top 5 concepts for space</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Integration Notes */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h2 className="text-lg font-bold mb-3">🔧 Integration Checklist</h2>
        <ul className="text-sm space-y-2">
          <li className="flex gap-2">
            <span>✅</span>
            <span>Update backend API to return citations in response</span>
          </li>
          <li className="flex gap-2">
            <span>✅</span>
            <span>Include decision_context in response payload</span>
          </li>
          <li className="flex gap-2">
            <span>✅</span>
            <span>Expose current_state and mastery_map in session response</span>
          </li>
          <li className="flex gap-2">
            <span>✅</span>
            <span>Parse and pass state_history from session state</span>
          </li>
          <li className="flex gap-2">
            <span>⏳</span>
            <span>Update tutor page component to use new types</span>
          </li>
          <li className="flex gap-2">
            <span>⏳</span>
            <span>Add user settings panel for citation preferences</span>
          </li>
        </ul>
      </div>
    </div>
  );
}

