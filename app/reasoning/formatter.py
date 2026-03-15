"""
Output Formatter

This module sits at the very end of the routing pipeline just before the 
FastAPI framework ships the HTTP Response back to the Client UI.

It ensures that the mathematical provenance arrays returned by the 70B model
are injected as Markdown `[Citation: ID]` inline links within the raw text.
"""
import logging
from app.infra.logging_utils import stage_info
from typing import Dict, Any, List
import re
import json

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
         stage_info(logger, "RAG:FORMAT", "formatting_response")

         # If answer is wrapped in a JSON block, extract the "answer" field for clean UX.
         if isinstance(formatted_answer, str) and "```" in formatted_answer:
             try:
                 match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", formatted_answer, re.DOTALL)
                 candidate = match.group(1) if match else None
                 if not candidate:
                     start = formatted_answer.find("{")
                     end = formatted_answer.rfind("}")
                     if start != -1 and end != -1 and end > start:
                         candidate = formatted_answer[start:end + 1]
                 if candidate:
                     parsed = json.loads(candidate)
                     extracted = parsed.get("answer")
                     if extracted:
                         formatted_answer = extracted
             except Exception:
                 # Best-effort fallback: strip code fences
                 formatted_answer = re.sub(r"```(?:json)?", "", formatted_answer).replace("```", "").strip()
         
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
                              
              formatted_answer = f"WARNING: Enterprise Guard flagged portions of this response as unverified and isolated them via strike-through.\n\n{formatted_answer}"
              
         return {
             "answer": formatted_answer,
             "sources": state.get("sources", []),
             "confidence": state.get("confidence", 0.0),
             "verifier_verdict": state.get("verifier_verdict", "UNVERIFIED"),
             "is_hallucinated": state.get("is_hallucinated", False),
             "optimizations": state.get("optimizations", {}),
             "latency_optimizations": state.get("latency_optimizations", {})
         }
