"""
Output Formatter

This module sits at the very end of the routing pipeline just before the 
FastAPI framework ships the HTTP Response back to the Client UI.

It ensures that the mathematical provenance arrays returned by the 70B model
are injected as Markdown `[Citation: ID]` inline links within the raw text.
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """
    Standardizes the LLM JSON Outputs into clean Markdown syntax for Streamlit/React.
    """
    
    @staticmethod
    def construct_final_response(state: Dict[str, Any]) -> Dict[str, Any]:
         """
         Prepares the final dictionary for the FastAPI Route layer schema `ChatResponse`.
         """
         # Citations are injected natively in Synthesis wrapper, we pull the direct answer here
         formatted_answer = state.get("answer", "")
         claims = state.get("verification_claims", [])
         
         # Append visual warning if the Independent Verifier caught a logic flaw
         if state.get("is_hallucinated", False):
              logger.warning("[FORMATTER] Hallucination detected. Striking out unsupported claims visually.")
              
              if claims:
                  for claim_obj in claims:
                      if claim_obj.get("verdict") == "UNSUPPORTED":
                          bad_sentence = claim_obj.get("claim", "")
                          # Execute structural redaction exactly on the hallucinated node
                          if bad_sentence in formatted_answer:
                              # Use Markdown strike-through
                              formatted_answer = formatted_answer.replace(bad_sentence, f"~~{bad_sentence}~~")
                              
              formatted_answer = f"⚠️ **Enterprise Guard Warning: The logic engine has flagged portions of this response as unverified and mathematically isolated them via strike-through.**\n\n{formatted_answer}"
              
         return {
             "answer": formatted_answer,
             "sources": state.get("sources", []),
             "confidence": state.get("confidence", 0.0),
             "verifier_verdict": state.get("verifier_verdict", "UNVERIFIED"),
             "is_hallucinated": state.get("is_hallucinated", False),
             "optimizations": state.get("optimizations", {}),
             "latency_optimizations": state.get("latency_optimizations", {})
         }
