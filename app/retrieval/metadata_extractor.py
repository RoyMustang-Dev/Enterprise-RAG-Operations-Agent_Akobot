"""
Dynamic Metadata Extraction Engine

This component replaces hardcoded filename filters. It uses a specialized reasoning model 
(`qwen-32b` or `qwen2.5-coder-32b`) to mathematically parse the user's intent and 
generate a strict JSON schema corresponding to Vector Database `$eq` and `$in` filters.

By applying these filters BEFORE the semantic search, we drastically reduce hallucinations.
"""
import os
import json
import logging
from typing import Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt
import aiohttp

logger = logging.getLogger(__name__)

class MetadataExtractor:
    """
    Translates Ambiguous Human Queries -> Strict Qdrant Filter Schemas.
    """
    
    def __init__(self, model_override: str = "llama-3.1-8b-instant"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_id = model_override
        
        # We explicitly supply the metadata fields the database actually possesses
        self.available_fields = [
            "document_type",   # e.g., 'pdf', 'docx', 'webpage'
            "source_domain",   # e.g., 'github.com', 'internal_wiki'
            "author", 
            "creation_year",
            "source"
        ]
        
        self.system_prompt = f'''SYSTEM: You are a high-precision metadata extractor. Given the USER QUERY and the AVAILABLE_METADATA_FIELDS list, extract structured filters as JSON following this EXACT schema:

{{
  "filters": {{
     "<field_name>": {{"op":"$eq" | "$in", "value": <string|array>}}
  }},
  "confidence": 0.00-1.00,
  "extracted_from": "<which phrase in user prompt, <=50 chars>"
}}

AVAILABLE_METADATA_FIELDS: {self.available_fields}

Rules:
- Only include fields explicitly present in AVAILABLE_METADATA_FIELDS.
- If the user does not specify a field organically, omit it. Do not guess.
- Example: "Show 2022 revenue from the wiki" -> {{"filters": {{"creation_year": {{"op": "$eq", "value": "2022"}}, "source_domain": {{"op": "$eq", "value": "internal_wiki"}}}}}}
'''

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
        if not self.api_key:
             logger.warning("[METADATA] GROQ_API_KEY missing. Returning empty filter bounds.")
             return {"filters": {}, "confidence": 0.0, "extracted_from": "bypassed"}
             
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0, # JSON requires complete determinism
            "max_tokens": 300,
            "response_format": {"type": "json_object"}
        }
        
        try:
            logger.info(f"[METADATA] Extracting filters dynamically via {self.model_id}...")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=10
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
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
                        if field in self.available_fields and isinstance(config, dict) and "op" in config and "value" in config:
                            safe_filters[field] = config
                        else:
                            logger.warning(f"[METADATA] Purged Hallucinated Schema Boundary: {field}")
                            
                    result["filters"] = safe_filters
                    
                    if safe_filters:
                         logger.info(f"[METADATA] Successfully parsed dynamic boundaries: {safe_filters}")
                    
                    return result
            
        except Exception as e:
             logger.error(f"[METADATA] Failure extracting Schema JSON. Reverting to unfiltered semantic wide-search: {e}")
             return {"filters": {}, "confidence": 0.0, "extracted_from": "Error Exception Fallback"}
