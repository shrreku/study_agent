'use client';

import React from 'react';

interface MasteryBarProps {
  concept: string;
  level: number;  // 0-1
}

/**
 * Displays mastery level for a single concept as a progress bar
 */
export function MasteryBar({ concept, level }: MasteryBarProps) {
  const percentage = Math.round(Math.max(0, Math.min(1, level)) * 100);

  const getColor = (pct: number): string => {
    if (pct >= 80) return 'bg-green-500';
    if (pct >= 60) return 'bg-yellow-500';
    if (pct >= 40) return 'bg-orange-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs mb-1">
        <span className="font-medium text-gray-700">{concept}</span>
        <span className="text-gray-500">{percentage}%</span>
      </div>
      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full ${getColor(percentage)} transition-all duration-500`}
          style={{ width: `${percentage}%` }}
          role="progressbar"
          aria-valuenow={percentage}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${concept} mastery level`}
        />
      </div>
    </div>
  );
}

