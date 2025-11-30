"""
Tutor Policies for MDP Level 3

This module implements different policy strategies for the Tutor MDP:
1. RuleBasedTutorPolicy - Deterministic rules for baseline
2. LLMTutorPolicy - LLM-based policy with structured output
3. TrainableTutorPolicy - Interface for RL-trained policies

The policy takes a TutorObservation and outputs a TutorAction.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import prompts
from mdp.schemas_v2 import (
    TutorPolicy, 
    TutorObservation, 
    TutorAction, 
    PedagogicalAction,
    StudentIntent,
    CorrectnessLevel,
)

if TYPE_CHECKING:
    from mdp.llm_client import LLMClient

logger = logging.getLogger(__name__)


# =============================================================================
# RULE-BASED POLICY (Baseline)
# =============================================================================

class RuleBasedTutorPolicy(TutorPolicy):
    """
    Deterministic rule-based policy for baseline comparison.
    
    Decision tree that RESPECTS recommended_action from InputAnalyzer,
    then selects appropriate pedagogical action.
    """
    
    def select_action(self, observation: TutorObservation) -> TutorAction:
        """Select action using hand-crafted rules, respecting recommended_action"""
        intent = observation.student_intent
        correctness = observation.student_correctness
        consecutive_wrong = observation.consecutive_incorrect
        hints_given = observation.hints_given
        mastery = observation.mastery_current
        recommended = observation.recommended_action  # Key: respect this!
        
        action = PedagogicalAction.EXPLAIN  # Default
        thinking = f"Rule-based: intent={intent.value}, correctness={correctness.value}, recommended={recommended}"
        
        # =================================================================
        # PRIORITY 1: Respect recommended_action for flow control
        # =================================================================
        if recommended == "advance":
            # Student should advance (correct answer OR explicit continue)
            if observation.step_index + 1 < observation.steps_total:
                action = PedagogicalAction.ADVANCE_STEP
                thinking += " -> Advancing to next step (recommended)"
            else:
                action = PedagogicalAction.SUMMARIZE
                thinking += " -> Last step, summarizing concept"
            
            logger.info(f"rule_policy_decision action={action.value}")
            return TutorAction(action=action, thinking=thinking, confidence=0.9)
        
        elif recommended == "replan":
            action = PedagogicalAction.REPLAN
            thinking += " -> Replanning (student is lost)"
            logger.info(f"rule_policy_decision action={action.value}")
            return TutorAction(action=action, thinking=thinking, confidence=0.85)
        
        # =================================================================
        # PRIORITY 2: For "reply" and "stay", pick best pedagogical action
        # =================================================================
        
        # Handle explicit continue intent (might have recommended="stay" but intent says continue)
        if intent == StudentIntent.CONTINUE:
            if observation.step_index + 1 < observation.steps_total:
                action = PedagogicalAction.ADVANCE_STEP
                thinking += " -> Explicit continue, advancing"
            else:
                action = PedagogicalAction.SUMMARIZE
                thinking += " -> Last step, summarizing"
        
        elif intent == StudentIntent.ANSWER:
            if correctness == CorrectnessLevel.CORRECT:
                # Should have been caught by recommended="advance" but fallback
                if observation.step_index + 1 < observation.steps_total:
                    action = PedagogicalAction.ADVANCE_STEP
                    thinking += " -> Correct answer, advancing"
                else:
                    action = PedagogicalAction.SUMMARIZE
                    thinking += " -> Correct answer, last step"
            elif correctness == CorrectnessLevel.PARTIAL:
                if hints_given < 2:
                    action = PedagogicalAction.GIVE_HINT
                    thinking += " -> Partial answer, giving hint"
                else:
                    action = PedagogicalAction.EXPLAIN
                    thinking += " -> Partial answer, explaining more"
            else:  # Incorrect
                if consecutive_wrong >= 2:
                    action = PedagogicalAction.EXPLAIN
                    thinking += " -> Multiple wrong, explaining directly"
                elif hints_given < 3:
                    action = PedagogicalAction.GIVE_HINT
                    thinking += " -> Wrong answer, giving hint"
                else:
                    action = PedagogicalAction.CORRECT_MISCONCEPTION
                    thinking += " -> Correcting misconception"
        
        elif intent == StudentIntent.QUESTION:
            action = PedagogicalAction.EXPLAIN
            thinking += " -> Student question, explaining"
        
        elif intent == StudentIntent.CONFUSION:
            if mastery < 0.3:
                action = PedagogicalAction.USE_ANALOGY
                thinking += " -> Confusion + low mastery, using analogy"
            else:
                action = PedagogicalAction.GIVE_HINT
                thinking += " -> Confusion, giving hint"
        
        elif intent == StudentIntent.ACKNOWLEDGE:
            # For "reply" recommended: respond with content
            # For "stay" recommended: also respond with content
            if recommended == "reply":
                # Student needs a response (e.g., answered a question)
                action = PedagogicalAction.EXPLAIN
                thinking += " -> Acknowledge + reply recommended, explaining"
            else:
                # Default acknowledge behavior: advance if possible
                if observation.step_index + 1 < observation.steps_total:
                    action = PedagogicalAction.ADVANCE_STEP
                    thinking += " -> Acknowledge, advancing to next step"
                else:
                    action = PedagogicalAction.SUMMARIZE
                    thinking += " -> Acknowledge on last step, summarizing"
        
        elif intent == StudentIntent.GIVE_UP:
            action = PedagogicalAction.WORKED_EXAMPLE
            thinking += " -> Student giving up, showing worked example"
        
        elif intent == StudentIntent.OFF_TOPIC:
            action = PedagogicalAction.NUDGE
            thinking += " -> Off-topic, nudging back"
        
        logger.info(f"rule_policy_decision action={action.value}")
        
        return TutorAction(
            action=action,
            thinking=thinking,
            confidence=0.8,
        )
    
    def generate_response(self, observation: TutorObservation, 
                          action: PedagogicalAction,
                          retrieved_context: str = "") -> str:
        """Rule-based policy doesn't generate responses - use ResponseGenerator"""
        return ""


# =============================================================================
# LLM-BASED POLICY (For Training)
# =============================================================================

class LLMTutorPolicy(TutorPolicy):
    """
    LLM-based policy that uses prompts to select actions.
    
    This is the policy architecture we train with PPO.
    It outputs structured decisions with thinking traces.
    """
    
    def __init__(self, llm_client: "LLMClient"):
        self.llm = llm_client
    
    def select_action(self, observation: TutorObservation) -> TutorAction:
        """
        Use LLM to select pedagogical action given observation.
        
        The model outputs:
        <think>reasoning</think>
        [Action: ACTION_NAME]
        """
        template = prompts.get("tutor_rl.policy_select_action")
        
        if not template:
            logger.warning("Policy prompt not found, falling back to rule-based")
            return RuleBasedTutorPolicy().select_action(observation)
        
        # Build prompt variables
        vars = {
            "observation": observation.to_prompt_string(),
            "available_actions": self._get_available_actions_str(observation),
            "action_descriptions": self._get_action_descriptions(),
        }
        
        user_prompt = prompts.render(template, vars)
        system_prompt = "You are the decision-making component of an intelligent tutoring system."
        
        logger.debug(f"policy_prompt_length={len(user_prompt)}")
        
        try:
            # Use raw completion to preserve thinking format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm.chat_completion(messages, max_tokens=400, temperature=0.3)
            
            if response and "choices" in response:
                raw_output = response["choices"][0]["message"]["content"]
                action = TutorAction.from_raw_output(raw_output)
                
                logger.info(f"llm_policy_decision action={action.action.value}", extra={
                    "observation_summary": {
                        "concept": observation.concept_id,
                        "intent": observation.student_intent.value,
                        "correctness": observation.student_correctness.value,
                    },
                    "selected_action": action.action.value,
                    "thinking_preview": action.thinking[:100] if action.thinking else "",
                    "confidence": action.confidence,
                })
                
                return action
            
        except Exception as e:
            logger.error(f"LLM policy failed: {e}", exc_info=True)
        
        # Fallback to rule-based
        logger.warning("LLM policy failed, using rule-based fallback")
        return RuleBasedTutorPolicy().select_action(observation)
    
    def generate_response(self, observation: TutorObservation, 
                          action: PedagogicalAction,
                          retrieved_context: str = "") -> str:
        """
        Generate response text for the selected action.
        
        Uses action-specific prompts from baseline.yaml.
        """
        # Map action to prompt key
        prompt_key = self._get_prompt_key_for_action(action)
        template = prompts.get(prompt_key)
        
        if not template:
            logger.warning(f"No prompt for action {action.value}, using explain")
            template = prompts.get("tutor.explain")
        
        vars = {
            "concept": observation.concept_id,
            "student_message": observation.student_message,
            "level": self._get_difficulty_level(observation.mastery_current),
            "context": retrieved_context,
            "step_goal": observation.step_subgoal or observation.step_content,
            "recent_errors": ", ".join(observation.recent_errors) if observation.recent_errors else "none",
            "mastery": f"{observation.mastery_current:.2f}",
        }
        
        user_prompt = prompts.render(template, vars)
        system_prompt = "You are an adaptive AI tutor."
        
        try:
            response_json = self.llm.call_json(system_prompt, user_prompt)
            
            # Extract response text
            text = (
                response_json.get("response") or 
                response_json.get("question") or 
                response_json.get("hint") or
                str(response_json)
            )
            
            logger.info(f"generated_response action={action.value} len={len(text)}")
            return text
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"Let me help you understand {observation.concept_id}."
    
    def _get_available_actions_str(self, observation: TutorObservation) -> str:
        """Get string listing available actions for current context"""
        # All content actions are available, but we highlight contextually relevant ones
        all_actions = PedagogicalAction.content_actions()
        
        lines = []
        for action in all_actions:
            marker = ""
            # Highlight likely good choices
            if observation.student_correctness == CorrectnessLevel.INCORRECT:
                if action in [PedagogicalAction.GIVE_HINT, PedagogicalAction.CORRECT_MISCONCEPTION]:
                    marker = " (recommended for incorrect answer)"
            elif observation.student_correctness == CorrectnessLevel.CORRECT:
                if action in [PedagogicalAction.REFLECT, PedagogicalAction.CHALLENGE]:
                    marker = " (recommended for correct answer)"
            elif observation.student_intent == StudentIntent.CONFUSION:
                if action in [PedagogicalAction.USE_ANALOGY, PedagogicalAction.EXPLAIN]:
                    marker = " (recommended for confusion)"
            
            lines.append(f"- {action.value}{marker}")
        
        return "\n".join(lines)
    
    def _get_action_descriptions(self) -> str:
        """Get descriptions of each action"""
        descriptions = {
            PedagogicalAction.SOCRATIC_QUESTION: "Ask a guiding question to help student discover the answer",
            PedagogicalAction.GIVE_HINT: "Provide partial information without revealing the full answer",
            PedagogicalAction.WORKED_EXAMPLE: "Walk through a similar problem step-by-step",
            PedagogicalAction.USE_ANALOGY: "Connect concept to something familiar",
            PedagogicalAction.NUDGE: "Gentle redirection when student is close",
            PedagogicalAction.EXPLAIN: "Clear, direct explanation of the concept",
            PedagogicalAction.CORRECT_MISCONCEPTION: "Address a specific error in understanding",
            PedagogicalAction.ELABORATE: "Add depth or additional detail",
            PedagogicalAction.SUMMARIZE: "Consolidate what's been learned",
            PedagogicalAction.CONCEPT_CHECK: "Quick question to verify understanding",
            PedagogicalAction.CHALLENGE: "Harder problem to test mastery",
            PedagogicalAction.REFLECT: "Prompt for metacognition",
        }
        
        lines = []
        for action, desc in descriptions.items():
            lines.append(f"- {action.value}: {desc}")
        
        return "\n".join(lines)
    
    def _get_prompt_key_for_action(self, action: PedagogicalAction) -> str:
        """Map action to prompt key"""
        mapping = {
            PedagogicalAction.SOCRATIC_QUESTION: "tutor_rl.action_socratic",
            PedagogicalAction.GIVE_HINT: "tutor_rl.action_hint",
            PedagogicalAction.WORKED_EXAMPLE: "tutor_rl.action_example",
            PedagogicalAction.USE_ANALOGY: "tutor_rl.action_analogy",
            PedagogicalAction.NUDGE: "tutor_rl.action_nudge",
            PedagogicalAction.EXPLAIN: "tutor.explain",
            PedagogicalAction.CORRECT_MISCONCEPTION: "tutor_rl.action_correct",
            PedagogicalAction.ELABORATE: "tutor.explain",
            PedagogicalAction.SUMMARIZE: "tutor_rl.action_summarize",
            PedagogicalAction.CONCEPT_CHECK: "tutor_rl.action_concept_check",
            PedagogicalAction.CHALLENGE: "tutor_rl.action_challenge",
            PedagogicalAction.REFLECT: "tutor.reflect",
        }
        return mapping.get(action, "tutor.explain")
    
    def _get_difficulty_level(self, mastery: float) -> str:
        """Map mastery to difficulty level"""
        if mastery < 0.3:
            return "beginner"
        elif mastery < 0.6:
            return "intermediate"
        else:
            return "advanced"


# =============================================================================
# HYBRID POLICY (Production)
# =============================================================================

class HybridTutorPolicy(TutorPolicy):
    """
    Hybrid policy that combines rule-based and LLM-based decisions.
    
    RESPECTS recommended_action for flow control, uses LLM for content actions.
    Good for production where we want reliability + flexibility.
    """
    
    def __init__(self, llm_client: Optional["LLMClient"] = None):
        self.llm_policy = LLMTutorPolicy(llm_client) if llm_client else None
        self.rule_policy = RuleBasedTutorPolicy()
    
    def select_action(self, observation: TutorObservation) -> TutorAction:
        """Select action using hybrid approach, respecting recommended_action"""
        
        recommended = observation.recommended_action
        intent = observation.student_intent
        correctness = observation.student_correctness
        
        # =================================================================
        # PRIORITY 1: Respect recommended_action for flow control
        # =================================================================
        if recommended == "advance":
            if observation.step_index + 1 < observation.steps_total:
                return TutorAction(
                    action=PedagogicalAction.ADVANCE_STEP,
                    thinking="Advancing (recommended_action=advance)",
                    confidence=0.95,
                )
            else:
                return TutorAction(
                    action=PedagogicalAction.SUMMARIZE,
                    thinking="Last step, summarizing (recommended_action=advance)",
                    confidence=0.9,
                )
        
        if recommended == "replan":
            return TutorAction(
                action=PedagogicalAction.REPLAN,
                thinking="Replanning (recommended_action=replan)",
                confidence=0.85,
            )
        
        # =================================================================
        # PRIORITY 2: Handle clear-cut intent cases
        # =================================================================
        
        # Case: Student giving up → worked example
        if intent == StudentIntent.GIVE_UP:
            return TutorAction(
                action=PedagogicalAction.WORKED_EXAMPLE,
                thinking="Student giving up, showing example (rule)",
                confidence=0.9,
            )
        
        # Case: Off-topic → nudge back
        if intent == StudentIntent.OFF_TOPIC:
            return TutorAction(
                action=PedagogicalAction.NUDGE,
                thinking="Off-topic, nudging back (rule)",
                confidence=0.85,
            )
        
        # Case: Explicit continue intent
        if intent == StudentIntent.CONTINUE:
            if observation.step_index + 1 < observation.steps_total:
                return TutorAction(
                    action=PedagogicalAction.ADVANCE_STEP,
                    thinking="Explicit continue, advancing (rule)",
                    confidence=0.95,
                )
        
        # =================================================================
        # PRIORITY 3: Nuanced cases - use LLM if available, else rules
        # =================================================================
        if self.llm_policy:
            try:
                return self.llm_policy.select_action(observation)
            except Exception as e:
                logger.warning(f"LLM policy failed, using rules: {e}")
        
        # Fallback to rules
        return self.rule_policy.select_action(observation)
    
    def generate_response(self, observation: TutorObservation, 
                          action: PedagogicalAction,
                          retrieved_context: str = "") -> str:
        """Generate response using LLM"""
        if self.llm_policy:
            return self.llm_policy.generate_response(observation, action, retrieved_context)
        return ""


# =============================================================================
# UNIFIED LLM POLICY (Single Call for Analysis + Action)
# =============================================================================

class UnifiedLLMPolicy(TutorPolicy):
    """
    Unified policy that does analysis + action selection in ONE LLM call.
    
    More efficient and coherent than separate InputAnalyzer + Policy calls.
    Returns both the action AND the response in one go.
    """
    
    def __init__(self, llm_client: "LLMClient"):
        self.llm = llm_client
        self.rule_policy = RuleBasedTutorPolicy()  # Fallback
        self._last_response: str = ""  # Cache response from select_action
    
    def select_action(self, observation: TutorObservation) -> TutorAction:
        """
        Unified analysis + action selection in one LLM call.
        
        Also caches the generated response for use by generate_response().
        """
        template = prompts.get("tutor_rl.unified_analyze_and_act")
        
        if not template:
            logger.warning("Unified prompt not found, falling back to rule-based")
            return self.rule_policy.select_action(observation)
        
        # Build recent history string
        history_str = ""
        for turn in observation.recent_history[-4:]:
            role = "Student" if turn.get("role") == "student" else "Tutor"
            history_str += f"  {role}: {turn.get('content', '')[:100]}\n"
        
        # Build prompt variables
        vars = {
            "concept": observation.concept_id,
            "step_index": observation.step_index + 1,
            "steps_total": observation.steps_total,
            "step_pedagogy": observation.step_pedagogy,
            "step_goal": observation.step_subgoal or observation.step_content or "not specified",
            "mastery": f"{observation.mastery_current:.2f}",
            "target_mastery": f"{observation.mastery_target:.2f}",
            "hints_given": observation.hints_given,
            "turn_count": observation.turn_count,
            "student_message": observation.student_message,
            "recent_history": history_str or "(session start)",
            "context": observation.retrieved_context[:1500] if observation.retrieved_context else "(no context)",
        }
        
        user_prompt = prompts.render(template, vars)
        system_prompt = "You are an intelligent tutoring system. Return valid JSON only."
        
        try:
            response_json = self.llm.call_json(system_prompt, user_prompt)
            
            # Parse response
            analysis = response_json.get("analysis", {})
            action_str = response_json.get("action", "explain")
            thinking = response_json.get("thinking", "")
            response_text = response_json.get("response", "")
            
            # Convert action string to PedagogicalAction
            action = PedagogicalAction.from_string(action_str)
            
            # Cache the response for generate_response()
            self._last_response = response_text
            
            logger.info(f"unified_policy_decision action={action.value}", extra={
                "intent": analysis.get("intent"),
                "correctness": analysis.get("correctness"),
                "thinking_preview": thinking[:80] if thinking else "",
            })
            
            return TutorAction(
                action=action,
                thinking=thinking,
                response_text=response_text,
                confidence=0.85,
            )
            
        except Exception as e:
            logger.error(f"Unified policy failed: {e}", exc_info=True)
            return self.rule_policy.select_action(observation)
    
    def generate_response(self, observation: TutorObservation, 
                          action: PedagogicalAction,
                          retrieved_context: str = "") -> str:
        """
        Return cached response from select_action, or generate new one.
        """
        if self._last_response:
            response = self._last_response
            self._last_response = ""  # Clear cache
            return response
        
        # Fallback: generate using LLMTutorPolicy
        llm_policy = LLMTutorPolicy(self.llm)
        return llm_policy.generate_response(observation, action, retrieved_context)


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

class ConversationalTutorPolicy:
    """
    Legacy policy for backward compatibility with existing orchestrator.
    
    Wraps the new policy architecture in the old interface.
    """
    
    def __init__(self, llm_client: Optional["LLMClient"] = None):
        self.policy = HybridTutorPolicy(llm_client)
    
    def decide(self, state_t: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy interface: decide action from dict observation.
        
        Returns dict with 'action' and 'feedback' keys.
        """
        # Handle button-based observation (legacy)
        if "button" in observation:
            btn = observation["button"]
            if btn == "replan":
                return {"action": "replan", "reason": "student_requested"}
            elif btn == "continue":
                if state_t.get("current_step_index", 0) + 1 >= len(state_t.get("plan", {}).get("steps", [])):
                    return {"action": "finish"}
                return {"action": "continue"}
        
        # Handle analysis-based observation
        analysis = observation.get("analysis", {})
        intent = analysis.get("intent", "acknowledge")
        correctness = analysis.get("correctness", "")
        feedback = analysis.get("feedback", "")
        rec_action = analysis.get("recommended_action", "")
        
        # Priority to recommended_action
        if rec_action:
            action_map = {
                "advance": "continue",
                "reply": "reply_to_user",
                "replan": "replan",
                "stay": "stay",
            }
            return {"action": action_map.get(rec_action, "stay"), "feedback": feedback}
        
        # Fallback logic
        if intent == "answer":
            if "correct" in correctness.lower() and "incorrect" not in correctness.lower():
                return {"action": "continue", "feedback": feedback or "Correct!"}
            elif "incorrect" in correctness.lower():
                return {"action": "stay", "feedback": feedback or "Let's try again."}
            else:
                return {"action": "stay", "feedback": feedback}
        elif intent == "question":
            return {"action": "reply_to_user", "feedback": feedback}
        elif intent == "confusion":
            return {"action": "stay", "feedback": feedback or "Let me explain differently."}
        
        # Default
        plan_steps = state_t.get("plan", {})
        if hasattr(plan_steps, "steps"):
            plan_steps = plan_steps.steps
        elif isinstance(plan_steps, dict):
            plan_steps = plan_steps.get("steps", [])
        else:
            plan_steps = []
            
        if state_t.get("current_step_index", 0) + 1 >= len(plan_steps):
            return {"action": "finish"}
        
        return {"action": "continue", "feedback": feedback}


# =============================================================================
# SIMPLE LEGACY POLICY
# =============================================================================

class SimpleTutorPolicy:
    """Simple button-based policy for legacy compatibility"""
    
    def decide(self, state_t: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        btn = observation.get("button")
        if btn == "replan":
            return {"action": "replan", "reason": "student_requested"}
        
        plan = state_t.get("plan")
        steps = plan.steps if hasattr(plan, "steps") else []
        
        if state_t.get("current_step_index", 0) + 1 >= len(steps):
            return {"action": "finish"}
        
        return {"action": "continue"}
