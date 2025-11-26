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
