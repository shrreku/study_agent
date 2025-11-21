'use client';

import React, { useState } from 'react';
import { Citation } from '../../types/tutor';
import { CitationCard } from './CitationCard';

interface CitationsContainerProps {
  citations: Citation[];
  defaultExpanded?: boolean;
  onViewFull?: (chunkId: string) => void;
}

/**
 * Container for displaying all citations with expand/collapse functionality
 */
export function CitationsContainer({
  citations,
  defaultExpanded = false,
  onViewFull,
}: CitationsContainerProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  if (!citations || citations.length === 0) {
    return null;
  }

  return (
    <div className="mt-4 border-t border-gray-200 pt-3">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-sm font-semibold text-gray-700 hover:text-gray-900 mb-2 hover:bg-gray-100 px-2 py-1 rounded transition-colors"
        aria-expanded={isExpanded}
        aria-controls="citations-list"
      >
        <span>📚 Sources ({citations.length})</span>
        <span className="text-gray-400 transition-transform" style={{
          transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
        }}>
          ▼
        </span>
      </button>

      {isExpanded && (
        <div id="citations-list" className="space-y-2 mt-3">
          {citations.map((citation, idx) => (
            <CitationCard
              key={`${citation.chunk_id}-${idx}`}
              citation={citation}
              index={idx + 1}
              onViewFull={onViewFull}
            />
          ))}
        </div>
      )}
    </div>
  );
}

