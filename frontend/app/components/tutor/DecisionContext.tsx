'use client';

import React from 'react';
import { DecisionContext as DecisionContextType } from '../../types/tutor';

interface DecisionContextProps {
  context?: DecisionContextType;
  visible?: boolean;
}

/**
 * Displays the rationale and trigger for the tutor's action
 */
export function DecisionContext({ context, visible = true }: DecisionContextProps) {
  if (!context || !visible || !context.rationale) {
    return null;
  }

  const getTriggerLabel = (triggerType?: string): string => {
    switch (triggerType) {
      case 'state_transition':
        return '🔄 State Transition';
      case 'safety_gate':
        return '🛡️ Safety Check';
      case 'policy':
        return '📋 Policy';
      case 'heuristic':
        return '⚙️ Heuristic';
      default:
        return '❓ Automatic';
    }
  };

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-3">
      <div className="flex items-start gap-2">
        <span className="text-blue-600 mt-0.5 text-lg">💭</span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-blue-900 mb-1">
            Why this action?
          </p>
          <p className="text-sm text-blue-800 mb-2">{context.rationale}</p>
          <div className="flex items-center gap-2">
            {context.cause && (
              <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                Trigger: {context.cause}
              </span>
            )}
            {context.trigger_type && (
              <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                {getTriggerLabel(context.trigger_type)}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

