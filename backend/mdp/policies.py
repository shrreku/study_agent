from typing import Dict, Any
from mdp.schemas import TutorPolicy

class SimpleTutorPolicy(TutorPolicy):
    def decide(self, state_t: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        # observation: {'button': 'continue'|'replan'}
        btn = observation.get("button")
        if btn == "replan":
            return {"action":"replan", "reason":"student_requested"}
        else:
            # continue default
            if state_t["current_step_index"]+1 >= len(state_t["plan"].steps):
                return {"action":"finish"}
            return {"action":"continue"}

class ConversationalTutorPolicy(TutorPolicy):
    def decide(self, state_t: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        # observation: {'analysis': {...}, 'button': ...}
        
        # Support fallback to button if provided
        if "button" in observation:
            btn = observation["button"]
            if btn == "replan":
                return {"action":"replan", "reason":"student_requested"}
            # If it's just "continue", we fall through to conversational logic or default
        
        analysis = observation.get("analysis", {})
        intent = analysis.get("intent", "acknowledge")
        correctness = analysis.get("correctness")
        feedback = analysis.get("feedback", "")
        rec_action = analysis.get("recommended_action")

        # Priority to recommended_action if available
        if rec_action:
            if rec_action == "advance":
                return {"action": "continue", "feedback": feedback}
            elif rec_action == "reply":
                 return {"action": "reply_to_user", "feedback": feedback}
            elif rec_action == "replan":
                 return {"action": "replan", "reason": "analysis_recommendation"}
            elif rec_action == "stay":
                 return {"action": "stay", "feedback": feedback}

        # Fallback Logic
        if intent == "answer":
            if correctness == "correct":
                 return {"action": "continue", "feedback": feedback or "Correct!"}
            elif correctness == "incorrect":
                 return {"action": "stay", "feedback": feedback or "That's not quite right. Let's try again."}
            else: 
                 return {"action": "stay", "feedback": feedback or "Can you elaborate?"}
        
        elif intent == "question":
             return {"action": "reply_to_user", "feedback": feedback}
        
        elif intent == "confusion":
             return {"action": "stay", "feedback": feedback or "Let me explain this part again."}
             
        if state_t["current_step_index"]+1 >= len(state_t["plan"].steps):
            return {"action": "finish"}
            
        return {"action": "continue", "feedback": feedback}
