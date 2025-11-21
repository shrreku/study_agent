'use client';

import React, { useState } from 'react';
import { Citation, PEDAGOGY_ICONS } from '../../types/tutor';

interface CitationCardProps {
  citation: Citation;
  index: number;
  onViewFull?: (chunkId: string) => void;
  maxPreviewLength?: number;
}

/**
 * Displays a single citation/source with expandable snippet
 */
export function CitationCard({
  citation,
  index,
  onViewFull,
  maxPreviewLength = 150,
}: CitationCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const getPedagogyIcon = (role?: string): string => {
    return (
      PEDAGOGY_ICONS[role as keyof typeof PEDAGOGY_ICONS] || '📄'
    );
  };

  const getPedagogyLabel = (role?: string): string => {
    if (!role) return 'Reference';
    return role
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const displayText = citation.snippet_text || citation.snippet_raw || '';
  const needsTruncation = displayText.length > maxPreviewLength;
  const previewText = displayText.slice(0, maxPreviewLength);

  return (
    <div className="border border-gray-200 rounded-lg p-3 mb-2 bg-gray-50 hover:bg-gray-100 transition-colors">
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span className="text-lg flex-shrink-0">
            {getPedagogyIcon(citation.pedagogy_role)}
          </span>
          <div className="min-w-0 flex-1">
            <span className="font-medium text-sm text-gray-700">
              Source {index}
            </span>
            <span className="text-xs text-gray-500 ml-2">
              {getPedagogyLabel(citation.pedagogy_role)}
            </span>
          </div>
        </div>
        {citation.relevance_score !== undefined && (
          <span className="flex-shrink-0 text-xs text-gray-500 bg-white px-2 py-1 rounded">
            {Math.round(citation.relevance_score * 100)}% relevant
          </span>
        )}
      </div>

      <div className="text-sm text-gray-600 whitespace-pre-wrap break-words">
        {isExpanded ? (
          <div>
            <p>{displayText}</p>
            {onViewFull && (
              <button
                onClick={() => onViewFull(citation.chunk_id)}
                className="text-blue-600 hover:text-blue-800 hover:underline mt-2 font-medium text-xs"
                aria-label="View full context for this citation"
              >
                View full context →
              </button>
            )}
          </div>
        ) : (
          <div>
            <p>
              {previewText}
              {needsTruncation && '...'}
            </p>
            {needsTruncation && (
              <button
                onClick={() => setIsExpanded(true)}
                className="text-blue-600 hover:text-blue-800 hover:underline ml-1 font-medium text-xs mt-1"
                aria-label="Read more"
              >
                Read more
              </button>
            )}
          </div>
        )}
      </div>

      {citation.page_number && (
        <div className="text-xs text-gray-400 mt-2">
          📄 Page {citation.page_number}
          {citation.source_document && ` • ${citation.source_document}`}
        </div>
      )}
    </div>
  );
}

