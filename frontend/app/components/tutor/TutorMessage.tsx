'use client';

import React from 'react';
import { TutorResponse, DEFAULT_TUTOR_SETTINGS } from '../../types/tutor';
import { CitationsContainer } from './CitationsContainer';
import { DecisionContext } from './DecisionContext';

interface TutorMessageProps {
  message: TutorResponse;
  showCitations?: boolean;
  showDecisionContext?: boolean;
  debugMode?: boolean;
  onViewFullCitation?: (chunkId: string) => void;
}

/**
 * Complete tutor message with citations, decision context, and metadata
 */
export function TutorMessage({
  message,
  showCitations = DEFAULT_TUTOR_SETTINGS.showCitations,
  showDecisionContext = DEFAULT_TUTOR_SETTINGS.showDecisionContext,
  debugMode = DEFAULT_TUTOR_SETTINGS.debugMode,
  onViewFullCitation,
}: TutorMessageProps) {
  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 max-w-3xl">
      {/* Main response text */}
      <div className="prose prose-sm max-w-none mb-3">
        <p className="text-gray-900 whitespace-pre-wrap break-words">
          {message.response_text}
        </p>
      </div>

      {/* Decision Context (optional) */}
      {showDecisionContext && message.decision_context && (
        <DecisionContext context={message.decision_context} visible={true} />
      )}

      {/* Citations (optional) */}
      {showCitations && message.citations && message.citations.length > 0 && (
        <CitationsContainer
          citations={message.citations}
          onViewFull={onViewFullCitation}
        />
      )}

      {/* Legacy source chunks (for backwards compat, only in debug) */}
      {debugMode && message.source_chunk_ids && message.source_chunk_ids.length > 0 && (
        <div className="text-xs text-gray-400 mt-3 pt-3 border-t border-gray-200">
          Chunk IDs: {message.source_chunk_ids.map((id) => (
            <code key={id} className="bg-gray-100 px-1 py-0.5 rounded mr-1">
              {id}
            </code>
          ))}
        </div>
      )}

      {/* Debug metadata */}
      {debugMode && (
        <div className="text-xs text-gray-400 mt-3 pt-3 border-t border-gray-200 space-y-1">
          <div>
            <strong>Action:</strong> {message.action} | <strong>Confidence:</strong>{' '}
            {message.confidence.toFixed(2)}
          </div>
          {message.grounding_mode && (
            <div>
              <strong>Grounding:</strong> {message.grounding_mode}
            </div>
          )}
          {message.concept && (
            <div>
              <strong>Concept:</strong> {message.concept}
            </div>
          )}
          {message.level && (
            <div>
              <strong>Level:</strong> {message.level}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

