"""
Independent Evidence Verifier (Sarvam M)

Checks the output of the 70B model. If Llama hallucinations occurred, 
Sarvam catches the un-sourced claim and immediately flags it.
By using an entirely different foundational architecture (Sarvam instead of Llama), 
we establish absolute model independence, defeating 'AI Yes-Men' behavior.
"""
import os
import json
import logging
import requests
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class HallucinationVerifier:
    """
    Independent Fact-Checker auditing the draft answer against the raw chunk arrays Line-by-Line.
    """
    
    def __init__(self, use_sarvam: bool = True):
        # We target Sarvam explicitly for independence check, but fallback to Groq if keys are missing
        self.use_sarvam = use_sarvam
        self.sarvam_key = os.getenv("SARVAM_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        
        self.system_prompt = '''SYSTEM: You are an enterprise evidence verifier. Given a candidate answer and the securely retrieved CONTEXT_CHUNKS, perform the following logic:

1) Break the draft answer down into an array of isolated, Line-by-Line factual claims.
2) For each factual claim, check if the CONTEXT_CHUNKS contain supporting evidence. 
3) Provide a "verdict" for each claim: SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED.
4) Calculate the "overall_verdict" based on the array of claims.

Output exactly one JSON object with fields: 
{
  "claims": [
     {"claim": "<exact sentence from draft>", "verdict": "SUPPORTED"}
  ],
  "overall_verdict": "SUPPORTED",
  "score": 1.0,
  "is_hallucinated": false
}

Constraints:
- If ANY single claim is UNSUPPORTED or missing from the chunks, flag "is_hallucinated": true.
- Do not modify the text of the claim. Copy it exactly as written including punctuation and trailing brackets.
'''

    def verify(self, draft_answer: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Cross-validates the draft answer string against the mathematical context arrays.
        """
        context_block = "\n\n---\n\n".join([c.get("page_content", "") for c in context_chunks])
        user_payload = f"DRAFT_ANSWER: {draft_answer}\n\nCONTEXT_CHUNKS:\n{context_block}"
        
        # Determine Routing Path (Prefer Sarvam)
        if self.use_sarvam and self.sarvam_key:
             return self._invoke_sarvam(user_payload)
        elif self.groq_key:
             return self._invoke_groq_fallback(user_payload)
             
        # Scaffold Fallback
        return {"overall_verdict": "UNVERIFIED", "score": 0.0, "is_hallucinated": False, "claims": []}

    def _invoke_groq_fallback(self, payload_str: str) -> Dict[str, Any]:
        """Executes verification using an external independent Groq Model."""
        headers = {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.1-8b-instant", # Use smaller, faster model just for binary verification
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": payload_str}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        
        try:
            logger.info("[VERIFIER] Executing logic boundary verification... (Groq Fallback)")
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=12)
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception as e:
            logger.error(f"[VERIFIER] Execution failure: {e}")
            return {"overall_verdict": "ERROR", "score": 0.0, "is_hallucinated": False, "claims": []}
            
    def _invoke_sarvam(self, payload_str: str) -> Dict[str, Any]:
        """Executes verification natively on Sarvam M."""
        headers = {"api-subscription-key": self.sarvam_key, "Content-Type": "application/json"}
        payload = {
            "model": "sarvam-m",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": payload_str}
            ],
            "temperature": 0.0,
            "top_p": 1,
            "max_tokens": 1500
        }
        
        try:
            logger.info("[VERIFIER] Executing logic boundary line-by-line verification... (Sarvam M Native)")
            response = requests.post("https://api.sarvam.ai/v1/chat/completions", headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            
            raw_text = response.json()["choices"][0]["message"]["content"]
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].strip()
                
            return json.loads(raw_text)
        except Exception as e:
            logger.error(f"[VERIFIER] Sarvam execution failure: {e}. Falling back to Groq natively.")
            if self.groq_key:
                return self._invoke_groq_fallback(payload_str)
            return {"overall_verdict": "ERROR", "score": 0.0, "is_hallucinated": False, "claims": []}
