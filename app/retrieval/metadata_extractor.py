"""
Dynamic Metadata Extraction Engine

This component replaces hardcoded filename filters. It uses a specialized reasoning model 
(`qwen-32b` or `qwen2.5-coder-32b`) to mathematically parse the user's intent and 
generate a strict JSON schema corresponding to Vector Database `$eq` and `$in` filters.

By applying these filters BEFORE the semantic search, we drastically reduce hallucinations.
"""
import os
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from typing import Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt
from app.infra.llm_client import achat_completion
from app.infra.model_registry import get_phase_model

logger = logging.getLogger(__name__)

class MetadataExtractor:
    """
    Translates Ambiguous Human Queries -> Strict Qdrant Filter Schemas.
    """
    
    def __init__(self, model_override: str = None):
        phase = get_phase_model("metadata_extractor")
        self.provider = phase["provider"]
        self.model_id = model_override or phase["model"]
        self.temperature = phase.get("temperature", 0.0)
        self.max_tokens = phase.get("max_tokens", 300)
        
        # We explicitly supply the metadata fields the database actually possesses
        self.available_fields = [
            "document_type",   # e.g., 'pdf', 'docx', 'webpage'
            "source_domain",   # e.g., 'github.com', 'internal_wiki'
            "author", 
            "creation_year",
            "source"
        ]
        
        from app.prompt_engine.groq_prompts.config import get_compiled_prompt
        base_prompt = get_compiled_prompt("metadata_extractor", self.model_id)
        self.system_prompt = f"{base_prompt}\n\nAVAILABLE_METADATA_FIELDS: {self.available_fields}\n"

    @retry(wait=wait_exponential(multiplier=1, min=2, max=6), stop=stop_after_attempt(3))
    async def extract_filters(self, user_prompt: str) -> Dict[str, Any]:
        """
        Synthesizes the JSON Object mapping for the Qdrant Filter Builder.
        
        Args:
            user_prompt (str): The raw string extracted from the HTTP payload.
            
        Returns:
            dict: The strict structural payload. Returns empty filters on failure to
                  prevent completely breaking the retrieval sequence.
        """
        if not os.getenv("MODELSLAB_API_KEY") and not os.getenv("GROQ_API_KEY"):
             logger.warning("[METADATA] GROQ_API_KEY missing. Returning empty filter bounds.")
             return {"filters": {}, "confidence": 0.0, "extracted_from": "bypassed"}
        
        try:
            logger.info(f"[METADATA] Extracting filters dynamically via {self.model_id}...")
            data = await achat_completion(
                provider=self.provider,
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                timeout=10,
            )
            raw_content = data["choices"][0]["message"]["content"]
            
            try:
                result = json.loads(raw_content)
            except json.JSONDecodeError:
                # Fallback extraction in case model wraps payload in markdown backticks
                import re
                match = re.search(r"```(?:json)?\n(.*?)\n```", raw_content, re.DOTALL)
                if match:
                    result = json.loads(match.group(1))
                else:
                    result = {}
            
            extracted_filters = result.get("filters", {})
            
            # Explicit Validation: Purge hallucinated DB Fields proactively
            safe_filters = {}
            for field, config in extracted_filters.items():
                if field not in self.available_fields:
                    logger.warning(f"[METADATA] Purged Hallucinated Schema Boundary: {field}")
                    continue
                if not isinstance(config, dict):
                    logger.warning(f"[METADATA] Purged invalid filter config for field: {field}")
                    continue
                op = config.get("op")
                value = config.get("value")
                if op not in {"$eq", "$in"} or value is None:
                    logger.warning(f"[METADATA] Purged invalid op/value for field: {field}")
                    continue
                safe_filters[field] = {"op": op, "value": value}
                    
            result["filters"] = safe_filters
            
            if safe_filters:
                 logger.info(f"[METADATA] Successfully parsed dynamic boundaries: {safe_filters}")
            
            return result
            
        except Exception as e:
             logger.error(f"[METADATA] Failure extracting Schema JSON. Reverting to unfiltered semantic wide-search: {e}")
             return {"filters": {}, "confidence": 0.0, "extracted_from": "Error Exception Fallback"}
