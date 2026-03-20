"""
Independent Evidence Verifier (Sarvam M)

Checks the output of the 70B model. If Llama hallucinations occurred, 
Sarvam catches the un-sourced claim and immediately flags it.
By using an entirely different foundational architecture (Sarvam instead of Llama), 
we establish absolute model independence, defeating 'AI Yes-Men' behavior.
"""
import os
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from app.infra.logging_utils import stage_info, stage_warn
import requests
from app.infra.llm_client import chat_completion, extract_message_content
from app.infra.model_registry import get_phase_model
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class HallucinationVerifier:
    """
    Independent Fact-Checker auditing the draft answer against the raw chunk arrays Line-by-Line.
    """
    
    def __init__(self, use_sarvam: bool = False):
        # Primary verifier uses ModelsLab Dev Tier (Gemini) for reliability.
        # Groq is used only as last-resort fallback when ModelsLab is unavailable.
        self.use_sarvam = use_sarvam
        self.sarvam_key = os.getenv("SARVAM_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        phase = get_phase_model("hallucination_verifier")
        self.provider = phase["provider"]
        self.model_id = phase["model"]
        self.temperature = phase.get("temperature", 0.0)
        self.max_tokens = phase.get("max_tokens", 600)
        
        from app.prompt_engine.groq_prompts.config import get_compiled_prompt
        self.system_prompt = get_compiled_prompt("hallucination_verifier", "sarvam-m")

    def verify(self, draft_answer: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Cross-validates the draft answer string against the mathematical context arrays.
        """
        context_block = "\n\n---\n\n".join([c.get("page_content", "") for c in context_chunks])
        user_payload = f"DRAFT_ANSWER: {draft_answer}\n\nCONTEXT_CHUNKS:\n{context_block}"
        
        # Determine Routing Path (Prefer ModelsLab)
        if os.getenv("MODELSLAB_API_KEY"):
            return self._invoke_modelslab(user_payload)
        if self.use_sarvam and self.sarvam_key:
            return self._invoke_sarvam(user_payload)
        if self.groq_key:
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
            stage_info(logger, "RAG:VERIFY", "groq_fallback=true")
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=12)
            if response.status_code == 400:
                # Retry without response_format for models that reject it
                payload.pop("response_format", None)
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=12)
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            return self._normalize_output(raw)
        except Exception as e:
            stage_warn(logger, "RAG:VERIFY", f"fallback_error={e}")
            return {"overall_verdict": "ERROR", "score": 0.0, "is_hallucinated": False, "claims": []}

    def _invoke_modelslab(self, payload_str: str) -> Dict[str, Any]:
        try:
            stage_info(logger, "RAG:VERIFY", "modelslab=true")
            data = chat_completion(
                provider=self.provider,
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": payload_str},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                timeout=20,
            )
            raw = extract_message_content(data)
            if not raw:
                raise ValueError("empty_response")
            normalized = self._normalize_output(raw)
            # If modelslab returned truncated/invalid JSON, fall back to Groq without raising noisy errors
            if (
                normalized.get("overall_verdict") == "UNVERIFIED"
                and ("```json" in raw or raw.strip().startswith("{"))
                and self.groq_key
            ):
                return self._invoke_groq_fallback(payload_str)
            return normalized
        except Exception as e:
            stage_warn(logger, "RAG:VERIFY", f"modelslab_error={e}")
            if self.groq_key:
                return self._invoke_groq_fallback(payload_str)
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
            stage_info(logger, "RAG:VERIFY", "sarvam_native=true")
            response = requests.post("https://api.sarvam.ai/v1/chat/completions", headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            
            raw_text = response.json()["choices"][0]["message"]["content"]
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].strip()
                
            return self._normalize_output(raw_text)
        except Exception as e:
            stage_warn(logger, "RAG:VERIFY", f"sarvam_error={e} fallback_to_groq=true")
            if self.groq_key:
                return self._invoke_groq_fallback(payload_str)
            return {"overall_verdict": "ERROR", "score": 0.0, "is_hallucinated": False, "claims": []}

    def _normalize_output(self, raw_text: str) -> Dict[str, Any]:
        """
        Normalizes verifier JSON to the internal schema:
        {overall_verdict, score, is_hallucinated, claims}
        """
        try:
            cleaned = raw_text or ""
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].strip()
            # If model returned a bare verdict string, normalize it.
            if cleaned.strip().upper() in {"SUPPORTED", "UNSUPPORTED", "UNVERIFIED"}:
                verdict = cleaned.strip().upper()
                return {
                    "overall_verdict": verdict,
                    "score": 0.0,
                    "is_hallucinated": verdict == "UNSUPPORTED",
                    "claims": [],
                }
            data = json.loads(cleaned)
        except Exception:
            return {"overall_verdict": "UNVERIFIED", "score": 0.0, "is_hallucinated": False, "claims": []}

        # Support both {"hallucinated": bool, "unsupported_claims": []} and native schema
        hallucinated = data.get("hallucinated")
        if hallucinated is None:
            hallucinated = data.get("is_hallucinated", False)
        claims = data.get("unsupported_claims") or data.get("claims") or []
        verdict = data.get("overall_verdict")
        if not verdict:
            verdict = "UNSUPPORTED" if hallucinated else "SUPPORTED"
        return {
            "overall_verdict": verdict,
            "score": float(data.get("score", 0.0) or 0.0),
            "is_hallucinated": bool(hallucinated),
            "claims": claims,
        }
