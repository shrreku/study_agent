'use client';

import React from 'react';
import { TutorState, STATE_CONFIG, MasteryInfo } from '../../types/tutor';
import { MasteryBar } from './MasteryBar';

interface TutorStateHeaderProps {
  currentState: TutorState;
  masteryMap: MasteryInfo;
  mode: string;
  stateHistory?: TutorState[];
}

/**
 * Displays current learning state, mastery progress, and mode
 */
export function TutorStateHeader({
  currentState,
  masteryMap,
  mode,
  stateHistory,
}: TutorStateHeaderProps) {
  const stateConfig = STATE_CONFIG[currentState];
  const concepts = Object.entries(masteryMap)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  return (
    <div className="bg-white rounded-lg shadow-sm p-4 mb-4 border border-gray-100">
      {/* Mastery Progress */}
      {concepts.length > 0 && (
        <div className="mb-4 pb-4 border-b border-gray-200">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">
            📊 Learning Progress
          </h3>
          <div className="space-y-3">
            {concepts.map(([concept, level]) => (
              <MasteryBar key={concept} concept={concept} level={level} />
            ))}
          </div>
        </div>
      )}

      {/* Current State */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${stateConfig.color}`}>
            {stateConfig.icon} {stateConfig.label.toUpperCase()}
          </span>
          {stateHistory && stateHistory.length > 0 && (
            <span className="text-xs text-gray-500 ml-2">
              Path: {stateHistory.slice(-3).map((s) => STATE_CONFIG[s].icon).join(' → ')}
            </span>
          )}
        </div>
        <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
          {mode}
        </span>
      </div>
    </div>
  );
}

