"""
Core Synthesis Engine (Llama 70B & Groq Native Annotations)

This is the primary mathematical reasoning brain of the Enterprise RAG System.
It explicitly accepts the Top-5 strictly reranked Vector DB chunks and forces the
model to synthesize an answer bounded entirely by those chunks natively utilizing 
Groq's `documents` API array mapping.

It embeds exponential backoff protections and strictly budgets input tokens.
"""
import os
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from app.infra.logging_utils import stage_info, stage_warn
from app.infra.llm_client import chat_completion, achat_completion_stream
from app.infra.model_registry import get_phase_model
import tiktoken
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SynthesisEngine:
    """
    Executes the dense grounding logic utilizing Groq's largest available reasoning model.
    """
    
    def __init__(self, model_override: str = None):
        phase = get_phase_model("rag_synthesis")
        self.provider = phase["provider"]
        self.model_id = model_override or phase["model"]
        self.fallback_provider = phase.get("fallback_provider")
        self.groq_model = os.getenv("RAG_SYNTH_GROQ_MODEL", "llama-3.3-70b-versatile")
        self.gemini_model = os.getenv("RAG_SYNTH_GEMINI_MODEL", "gemini-2.5-flash")
        
        # Tokenizer initialization for boundary limits
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.MAX_INPUT_TOKENS = 6500 # Strict ceiling below the 8K bound
        
        # The rigid System Prompt
        from app.prompt_engine.groq_prompts.config import get_compiled_prompt
        self.system_prompt = get_compiled_prompt("rag_synthesis", self.model_id)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=8), stop=stop_after_attempt(2))
    def _execute_api_call(self, payload: dict) -> dict:
        """Executes the POST HTTP layer securely guarded by an exponential backoff decorator."""
        stage_info(logger, "RAG:SYNTH", "api_call")
        # 20 second timeout to avoid excessive tail latency on large models
        return chat_completion(
            provider=payload.get("provider", self.provider),
            model=payload["model"],
            messages=payload["messages"],
            temperature=payload.get("temperature", 0.3),
            max_tokens=payload.get("max_tokens"),
            response_format=payload.get("response_format"),
            timeout=20,
        )

    def synthesize(self, user_prompt: str, context_chunks: List[Dict[str, Any]],
                   override_model: str = None, override_temp: float = None) -> Dict[str, Any]:
        """
        Synthesizes the final conversational output while structurally balancing token economics.
        """
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
             return {"answer": "Error: API Key missing.", "provenance": [], "confidence": 0.0}
             
        # Target Model Assignments via Routing Override
        active_model = override_model if override_model else self.model_id
        active_temp = override_temp if override_temp is not None else 0.3
             
        # Token Budgeting
        system_overhead = len(self.tokenizer.encode(self.system_prompt))
        user_prompt_overhead = len(self.tokenizer.encode(user_prompt))
        budget_remaining = self.MAX_INPUT_TOKENS - (system_overhead + user_prompt_overhead)
        
        context_string = ""
        cost_sum = 0
        provenance = []
        
        # Concat context into string natively since Groq 70B no longer supports the schema
        for i, chunk in enumerate(context_chunks):
            chunk_text = chunk.get("page_content", "")
            cost = len(self.tokenizer.encode(chunk_text))
            
            if cost_sum + cost > budget_remaining:
                logger.warning(f"[SYNTHESIS] Truncating remaining chunks to survive Token Budget. Cost hit {cost_sum}/{budget_remaining}")
                break
                
            native_id = str(chunk.get("source", f"Chunk-{i}"))
            context_string += f"[Doc {native_id}]:\n{chunk_text}\n---\n"
            cost_sum += cost
            provenance.append(
                {
                    "source": native_id,
                    "score": chunk.get("score", 0.0),
                    "text": chunk_text[:240],
                }
            )
            
        payload = {
            "provider": self.provider,
            "model": active_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"CONTEXT DOCUMENTS:\n{context_string}\n\nUSER_PROMPT: {user_prompt}"}
            ],
            "temperature": active_temp, # Bounded creativity
            "max_tokens": 2048
        }
        
        try:
            stage_info(logger, "RAG:SYNTH", "payload_ready")
            response_json = self._execute_api_call(payload)
            raw_content = response_json["choices"][0]["message"]["content"]
            try:
                parsed_result = json.loads(raw_content)
                answer_text = parsed_result.get("answer", "")
                confidence = parsed_result.get("confidence", 0.0)
            except Exception:
                # Try to extract fenced JSON block or best-effort JSON slice
                try:
                    import re as _re
                    match = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, _re.DOTALL)
                    candidate = match.group(1) if match else None
                    if not candidate:
                        start = raw_content.find("{")
                        end = raw_content.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            candidate = raw_content[start:end + 1]
                    if candidate:
                        parsed_result = json.loads(candidate)
                        answer_text = parsed_result.get("answer", raw_content.strip())
                        confidence = parsed_result.get("confidence", 0.0)
                    else:
                        raise ValueError("no json candidate")
                except Exception:
                    # Last-resort: extract answer field from JSON-ish text.
                    try:
                        import re as _re
                        match = _re.search(r'"answer"\s*:\s*"(.*?)"\s*,\s*"confidence"', raw_content, _re.DOTALL)
                        if match:
                            candidate = match.group(1)
                            candidate = candidate.replace("\\n", "\n").replace("\\t", "\t").replace("\\\"", "\"")
                            answer_text = candidate.strip()
                            confidence = 0.0
                        else:
                            raise ValueError("no answer field")
                    except Exception:
                        # Fallback: treat raw content as the answer if JSON parsing fails
                        stage_warn(logger, "RAG:SYNTH", "non_json_output_fallback")
                        answer_text = raw_content.strip()
                        confidence = 0.0

            # Approx token counts (input from prompt/context, output from answer text)
            tokens_input = system_overhead + user_prompt_overhead + cost_sum
            tokens_output = len(self.tokenizer.encode(answer_text)) if answer_text else 0

            return {
                "answer": answer_text,
                "provenance": provenance,
                "confidence": confidence,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "temperature_used": active_temp,
                "model_used": active_model
            }
            
        except Exception as e:
             stage_warn(logger, "RAG:SYNTH", f"crash={e}")
             # Fallback chain: Groq -> Gemini
             fallback_chain = []
             if os.getenv("GROQ_API_KEY") and self.provider != "groq":
                 fallback_chain.append(("groq", self.groq_model))
             if os.getenv("GEMINI_API_KEY") and self.provider != "gemini":
                 fallback_chain.append(("gemini", self.gemini_model))
             for provider, model in fallback_chain:
                 try:
                     stage_warn(logger, "RAG:SYNTH", f"fallback={provider}:{model}")
                     payload_fb = dict(payload)
                     payload_fb["provider"] = provider
                     payload_fb["model"] = model
                     response_json = self._execute_api_call(payload_fb)
                     raw_content = response_json["choices"][0]["message"]["content"]
                     try:
                         parsed_result = json.loads(raw_content)
                         answer_text = parsed_result.get("answer", "")
                         confidence = parsed_result.get("confidence", 0.0)
                     except Exception:
                         answer_text = raw_content.strip()
                         confidence = 0.0
                     tokens_input = system_overhead + user_prompt_overhead + cost_sum
                     tokens_output = len(self.tokenizer.encode(answer_text)) if answer_text else 0
                     return {
                         "answer": answer_text,
                         "provenance": provenance,
                         "confidence": confidence,
                         "tokens_input": tokens_input,
                         "tokens_output": tokens_output,
                         "temperature_used": active_temp,
                         "model_used": model
                     }
                 except Exception as fb_err:
                     stage_warn(logger, "RAG:SYNTH", f"fallback_crash={fb_err}")
             # Extractive fallback to avoid hard failure
             fallback_bits = []
             for chunk in context_chunks[:3]:
                 text = (chunk.get("page_content") or "").strip()
                 if text:
                     fallback_bits.append(text[:400])
             if fallback_bits:
                 fallback_answer = "Extractive summary (LLM unavailable):\n\n" + "\n\n".join(fallback_bits)
                 return {
                     "answer": fallback_answer,
                     "provenance": provenance,
                     "confidence": 0.0,
                     "tokens_input": 0,
                     "tokens_output": 0,
                     "temperature_used": active_temp,
                     "model_used": active_model,
                 }
             return {
                 "answer": "Internal Generation Error validating tokens.",
                 "provenance": provenance,
                 "confidence": 0.0,
                 "tokens_input": 0,
                 "tokens_output": 0,
                 "temperature_used": active_temp,
                 "model_used": active_model,
             }

    async def synthesize_stream(
        self,
        user_prompt: str,
        context_chunks: List[Dict[str, Any]],
        override_model: str = None,
        override_temp: float = None,
        on_token=None,
    ) -> Dict[str, Any]:
        """
        Provider-native streaming synthesis. Streams tokens via callback while also
        returning the final aggregated response.
        """
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            return {"answer": "Error: API Key missing.", "provenance": [], "confidence": 0.0}

        active_model = override_model if override_model else self.model_id
        active_temp = override_temp if override_temp is not None else 0.3

        system_overhead = len(self.tokenizer.encode(self.system_prompt))
        user_prompt_overhead = len(self.tokenizer.encode(user_prompt))
        budget_remaining = self.MAX_INPUT_TOKENS - (system_overhead + user_prompt_overhead)

        context_string = ""
        cost_sum = 0
        provenance = []
        for i, chunk in enumerate(context_chunks):
            chunk_text = chunk.get("page_content", "")
            cost = len(self.tokenizer.encode(chunk_text))
            if cost_sum + cost > budget_remaining:
                logger.warning(f"[SYNTHESIS] Truncating remaining chunks to survive Token Budget. Cost hit {cost_sum}/{budget_remaining}")
                break
            native_id = str(chunk.get("source", f"Chunk-{i}"))
            context_string += f"[Doc {native_id}]:\n{chunk_text}\n---\n"
            cost_sum += cost
            provenance.append(
                {
                    "source": native_id,
                    "score": chunk.get("score", 0.0),
                    "text": chunk_text[:240],
                }
            )

        payload = {
            "model": active_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"CONTEXT DOCUMENTS:\n{context_string}\n\nUSER_PROMPT: {user_prompt}"}
            ],
            "temperature": active_temp,
            "max_tokens": 2048
        }

        answer_text = ""
        try:
            async for delta in achat_completion_stream(
                provider=self.provider,
                model=payload["model"],
                messages=payload["messages"],
                temperature=payload.get("temperature", 0.3),
                max_tokens=payload.get("max_tokens"),
                response_format=payload.get("response_format"),
                timeout=30,
            ):
                if delta and on_token:
                    on_token(delta)
                answer_text += delta
        except Exception as e:
            stage_warn(logger, "RAG:SYNTH", f"stream_crash={e}")
            # Fallback streaming to Groq/Gemini if primary fails
            fallback_chain = []
            if os.getenv("GROQ_API_KEY") and self.provider != "groq":
                fallback_chain.append(("groq", self.groq_model))
            if os.getenv("GEMINI_API_KEY") and self.provider != "gemini":
                fallback_chain.append(("gemini", self.gemini_model))
            for provider, model in fallback_chain:
                try:
                    async for delta in achat_completion_stream(
                        provider=provider,
                        model=model,
                        messages=payload["messages"],
                        temperature=payload.get("temperature", 0.3),
                        max_tokens=payload.get("max_tokens"),
                        response_format=payload.get("response_format"),
                        timeout=30,
                    ):
                        if delta and on_token:
                            on_token(delta)
                        answer_text += delta
                    break
                except Exception as fb_err:
                    stage_warn(logger, "RAG:SYNTH", f"stream_fallback_crash={fb_err}")

        # Parse JSON if possible (support fenced json blocks)
        confidence = 0.0
        try:
            parsed_result = json.loads(answer_text)
            answer_text = parsed_result.get("answer", "")
            confidence = parsed_result.get("confidence", 0.0)
        except Exception:
            try:
                import re as _re
                match = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", answer_text, _re.DOTALL)
                candidate = match.group(1) if match else None
                if not candidate:
                    start = answer_text.find("{")
                    end = answer_text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        candidate = answer_text[start:end + 1]
                if candidate:
                    parsed_result = json.loads(candidate)
                    answer_text = parsed_result.get("answer", answer_text)
                    confidence = parsed_result.get("confidence", 0.0)
            except Exception:
                # Last-resort extraction from JSON-ish stream
                try:
                    import re as _re
                    match = _re.search(r'"answer"\s*:\s*"(.*?)"\s*,\s*"confidence"', answer_text, _re.DOTALL)
                    if match:
                        candidate = match.group(1)
                        candidate = candidate.replace("\\n", "\n").replace("\\t", "\t").replace("\\\"", "\"")
                        answer_text = candidate.strip()
                        confidence = 0.0
                except Exception:
                    pass

        tokens_input = system_overhead + user_prompt_overhead + cost_sum
        tokens_output = len(self.tokenizer.encode(answer_text)) if answer_text else 0

        return {
            "answer": answer_text.strip(),
            "provenance": provenance,
            "confidence": confidence,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "temperature_used": active_temp,
            "model_used": active_model
        }
