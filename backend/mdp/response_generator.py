from typing import Dict, Any, Optional
import logging
import prompts
from mdp.llm_client import LLMClient
from mdp.rag import RAGTools

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, llm_client: LLMClient, rag_tools: RAGTools):
        self.llm = llm_client
        self.rag = rag_tools

    def generate_pedagogical_response(self, 
                                      step_concept: str, 
                                      step_pedagogy: str, 
                                      step_content: str, 
                                      student_history: list,
                                      plan_index: int = 0,
                                      plan_length: int = 1,
                                      feedback_override: Optional[str] = None,
                                      force_reply_mode: bool = False,
                                      student_mastery: float = 0.0,
                                      retrieval_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response using specific prompts from baseline.yaml.
        """
        # Map step_pedagogy to baseline prompt keys
        # Keys available in baseline.yaml: tutor.explain, tutor.ask, tutor.hint, tutor.reflect
        key_map = {
            "explain": "tutor.explain",
            "question": "tutor.ask",
            "hint": "tutor.hint",
            "reflect": "tutor.reflect",
            "example": "tutor.explain", # Reuse explain for examples for MVP simplicity
            "summary": "tutor.explain"  # Reuse explain for summaries
        }
        
        prompt_key = key_map.get(step_pedagogy.lower(), "tutor.explain")

        # If force_reply_mode, we are replying to a user query/action off-plan
        # We use tutor.explain as a generic "talk to student" prompt
        if force_reply_mode:
            prompt_key = "tutor.explain"

        template = prompts.get(prompt_key)

        if not template:
            return {"content": f"Error: Prompt {prompt_key} not found.", "error": "missing_prompt"}

        # Prepare Context
        # We format context into clearly labeled sections so prompts can
        # treat them as internal hints + reference material, not prior
        # conversational turns.
        step_label = f"PLAN_STEP_HINT (step {plan_index+1} of {plan_length}):"
        step_text = step_content or "(no specific hint provided)"

        combined_lines = [f"{step_label} {step_text}"]
        
        # Inject mastery info
        combined_lines.append(f"[STUDENT MASTERY]: {student_mastery:.2f} (0.0=novice, 1.0=expert)")

        if feedback_override:
            # feedback_override is an internal directive from the analyzer
            # indicating how the next response should behave.
            combined_lines.append(f"[INSTRUCTION]: {feedback_override}")

        # Smart RAG Query Strategy with multi-query support
        # 1. If retrieval_query is explicitly provided (from input analysis), use it.
        # 2. Else, construct 2-3 word queries from concept + pedagogy
        # 3. Fallback to just concept
        
        queries = []
        if retrieval_query:
            # If it's a list, use it directly
            if isinstance(retrieval_query, list):
                queries = retrieval_query
            elif isinstance(retrieval_query, str):
                # Split long queries into 2-3 word chunks
                words = retrieval_query.split()
                if len(words) > 3:
                    queries = [" ".join(words[i:i+2]) for i in range(0, min(6, len(words)), 2)]
                else:
                    queries = [retrieval_query]
        
        if not queries:
            # Fallback strategy: generate 2-3 word queries
            queries = [step_concept]
            if step_pedagogy and step_pedagogy.lower() not in ["explain", "intro"]:
                queries.append(f"{step_concept} {step_pedagogy}")

        # Use multi-query RAG for better coverage
        if len(queries) > 1:
            rag_context = self.rag.search_multi_query(queries, limit_per_query=2)
        else:
            rag_context = self.rag.search_context(queries[0] if queries else step_concept)
        
        if rag_context:
            combined_lines.append(f"REFERENCE_MATERIAL (searched for: {queries}):")
            combined_lines.append(str(rag_context))

        combined_context = "\n".join(combined_lines)

        # Format student history
        history_str = "None"
        if student_history:
             history_str = "\n".join([f"- {h.get('button', 'action')} at step {h.get('step_id')}" for h in student_history[-3:]])

        # Prepare variables for baseline prompts
        # They expect: {{concept}}, {{student_message}}, {{level}}, {{context}}
        # tutor.reflect also uses {{recent_history}}
        vars = {
            "concept": step_concept,
            "student_message": "continue", # Default since MDP drives the flow
            "level": "intermediate", # TODO: fetch from profile
            "context": combined_context,
            "recent_history": history_str
        }
        
        # Render
        user_prompt = prompts.render(template, vars)
        system_prompt = "You are an adaptive AI tutor."
        
        active_set = prompts.active_set()
        logger.info(f"generating_response_prompt key={prompt_key} set={active_set} len={len(user_prompt)}")

        # Call LLM
        try:
            response_json = self.llm.call_json(system_prompt, user_prompt)
            
            # Extract response
            # baseline prompts return {"response": ...} or {"question": ...}
            text_response = response_json.get("response") or response_json.get("question") or str(response_json)
            
            return {
                "content": text_response,
                "raw_json": response_json
            }
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {"content": step_content, "error": str(e)}
