/**
 * Type definitions for the Tutor Agent UI
 */

/**
 * Tutor state representing the learning phase
 */
export type TutorState = 
  | 'orientation' 
  | 'teaching' 
  | 'assessment' 
  | 'review' 
  | 'closure';

/**
 * Citation source for a tutor response
 */
export interface Citation {
  chunk_id: string;
  snippet_text: string;        // Cleaned text for display
  snippet_raw?: string;         // Original (not shown to user)
  relevance_score?: number;     // 0-1 confidence
  pedagogy_role?: 'definition' | 'example' | 'concept_check' | 'reference';
  page_number?: number;
  source_document?: string;
}

/**
 * Decision context explaining why an action was chosen
 */
export interface DecisionContext {
  rationale?: string;           // Human-readable explanation
  cause?: string;               // Trigger event (e.g., "3 consecutive explains")
  trigger_type?: 'state_transition' | 'safety_gate' | 'policy' | 'heuristic';
}

/**
 * Structured tutor response with metadata
 */
export interface TutorResponse {
  response_text: string;
  action: string;               // "explain", "ask", "hint", "reflect", "review"
  confidence: number;           // 0-1
  
  // NEW: Structured citations and context
  citations?: Citation[];
  grounding_mode?: 'llm_integrated' | 'explicit_citation' | 'none';
  decision_context?: DecisionContext;
  
  // Legacy fields (for backwards compatibility)
  source_chunk_ids?: string[];
  action_type?: string;
  concept?: string;
  level?: string;
  learning_path?: string[];
  cold_start?: boolean;
  intent?: string;
  affect?: string;
  classification_confidence?: number;
}

/**
 * Mastery level for a concept (0-1 scale)
 */
export interface MasteryInfo {
  [concept: string]: number;
}

/**
 * Complete session state
 */
export interface SessionState {
  current_state: TutorState;
  mastery_map: MasteryInfo;
  mode: string;                 // "simple", "intelligent", "step_by_step", "debug"
  state_history?: TutorState[];
}

/**
 * A single turn in the conversation
 */
export interface ConversationTurn {
  id: string;
  role: 'user' | 'tutor';
  content: string;
  meta?: {
    actionType?: string;
    confidence?: number;
    concept?: string;
    level?: string;
    learningPath?: string[];
    coldStart?: boolean;
    intent?: string;
    affect?: string;
    classificationConfidence?: number;
    sourceChunks?: string[];
    citations?: Citation[];
    decisionContext?: DecisionContext;
    state?: TutorState;
    masteryUpdate?: {
      concept: string;
      delta: number;
      reason: string;
    };
    raw?: Record<string, any>;
  };
}

/**
 * User preferences for tutor UI
 */
export interface TutorSettings {
  showCitations: boolean;           // Show/hide citation section
  expandCitationsDefault: boolean;  // Expand citations by default
  citationMaxLength: number;        // Characters to show before "Read more"
  showStateHeader: boolean;         // Show state and mastery bars
  showDecisionContext: boolean;     // Show decision rationale
  debugMode: boolean;               // Show raw metadata
}

/**
 * Default user settings
 */
export const DEFAULT_TUTOR_SETTINGS: TutorSettings = {
  showCitations: true,
  expandCitationsDefault: false,
  citationMaxLength: 150,
  showStateHeader: true,
  showDecisionContext: false,
  debugMode: false,
};

/**
 * Pedagogical tags for citations
 */
export const PEDAGOGY_ICONS: Record<string, string> = {
  definition: '📖',
  example: '💡',
  concept_check: '✓',
  reference: '📄',
};

/**
 * State visualization configuration
 */
export const STATE_CONFIG: Record<TutorState, { icon: string; color: string; label: string }> = {
  orientation: {
    icon: '🧭',
    color: 'bg-blue-100 text-blue-800',
    label: 'Getting Oriented',
  },
  teaching: {
    icon: '🎓',
    color: 'bg-green-100 text-green-800',
    label: 'Learning',
  },
  assessment: {
    icon: '📝',
    color: 'bg-yellow-100 text-yellow-800',
    label: 'Checking Understanding',
  },
  review: {
    icon: '🔄',
    color: 'bg-orange-100 text-orange-800',
    label: 'Reviewing',
  },
  closure: {
    icon: '✅',
    color: 'bg-gray-100 text-gray-800',
    label: 'Session Complete',
  },
};

