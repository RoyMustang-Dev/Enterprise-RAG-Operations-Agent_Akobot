import os
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

from app.prompt_engine.groq_prompts.base_prompts import GROQ_BASE_PROMPTS

class PersonaCacheManager:
    """
    Singleton cache to fetch the active Bootstrapper Persona exactly once
    to prevent SQLite 0ms overhead per API request.
    """
    _instance = None
    
    def __new__(cls, db_path="data/agent_profiles.db"):
        if cls._instance is None:
            cls._instance = super(PersonaCacheManager, cls).__new__(cls)
            cls._instance.db_path = db_path
            cls._instance._cached_persona = None
            cls._instance.refresh_cache()
        return cls._instance

    def refresh_cache(self):
        """Pulls the active persona from SQLite into RAM."""
        try:
            if not os.path.exists(self.db_path):
                self._cached_persona = None
                return
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Fetch the most recently saved agent persona
            cursor.execute("SELECT bot_name, brand_details, expanded_prompt, welcome_message FROM agent_profiles ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                self._cached_persona = {
                    "bot_name": row[0],
                    "brand_details": row[1],
                    "expanded_prompt": row[2],
                    "welcome_message": row[3] if len(row) > 3 else ""
                }
            else:
                self._cached_persona = None
            conn.close()
            logger.info("[PROMPT_ENGINE] Global Persona Cache Refreshed.")
        except Exception as e:
            logger.error(f"[PROMPT_ENGINE] Failed to read Persona SQLite Database: {e}")
            self._cached_persona = None

    def get_persona(self):
        return self._cached_persona


def get_compiled_prompt(stage: str, model: str) -> str:
    """
    Dynamically assembles the System Prompt by combining:
    1. The Global Persona (if applicable for the stage).
    2. The Base System String for the node.
    3. The Few-Shot JSON examples mapped explicitly to the passed `model` name.
    """
    # 1. Fetch the Base Template
    base_template = GROQ_BASE_PROMPTS.get(stage, "SYSTEM: You are a helpful assistant.")
    
    # 2. Fetch the Persona
    cache = PersonaCacheManager()
    persona = cache.get_persona()
    
    # Define which stages receive persona injection.
    # If PERSONA_INJECTION_MODE=all, all stages receive full injection.
    injection_mode = os.getenv("PERSONA_INJECTION_MODE", "strict").lower()
    FULL_INJECTION_STAGES = ["rag_synthesis", "multimodal_voice", "smalltalk_agent"]
    PARTIAL_INJECTION_STAGES = ["intent_classifier", "coder_agent"]
    NO_INJECTION_STAGES = [
        "security_guard",
        "query_rewriter",
        "metadata_extractor",
        "complexity_scorer",
        "reward_scorer",
        "hallucination_verifier",
    ]

    persona_max_chars = int(os.getenv("PERSONA_MAX_CHARS", "4000"))
    persona_partial_max_chars = int(os.getenv("PERSONA_PARTIAL_MAX_CHARS", "800"))
    
    persona_block = ""
    if injection_mode in {"off", "none", "disabled"}:
        persona = None
    if persona:
        expanded = (persona.get("expanded_prompt", "") or "").strip()
        if len(expanded) > persona_max_chars:
            expanded = expanded[:persona_max_chars]

        if injection_mode == "all":
            persona_block = f"""[GLOBAL PERSONA INITIATED]
You are dynamically mapped to "{persona.get('bot_name', 'Agent')}".
Brand Details: {persona.get('brand_details', '')}
Brand Welcome Greeting: {persona.get('welcome_message', '')}
Core Directives (ReAct/CoT/ToT Constraints):
{expanded}
[END GLOBAL PERSONA]

---
"""
        elif stage in FULL_INJECTION_STAGES:
            persona_block = f"""[GLOBAL PERSONA INITIATED]
You are dynamically mapped to "{persona.get('bot_name', 'Agent')}".
Brand Details: {persona.get('brand_details', '')}
Brand Welcome Greeting: {persona.get('welcome_message', '')}
Core Directives (ReAct/CoT/ToT Constraints):
{expanded}
[END GLOBAL PERSONA]

---
"""
        elif stage in PARTIAL_INJECTION_STAGES:
            # Strip heavy reasoning directives for small-model JSON tasks
            short_brand = (persona.get("brand_details", "") or "").strip()
            if len(short_brand) > persona_partial_max_chars:
                short_brand = short_brand[:persona_partial_max_chars]
            persona_block = f"""[GLOBAL PERSONA INITIATED]
You are dynamically mapped to "{persona.get('bot_name', 'Agent')}".
Brand Details: {short_brand}
[END GLOBAL PERSONA]

---
"""
        elif stage in NO_INJECTION_STAGES and injection_mode == "strict":
            persona_block = ""

    # 3. Assemble Few-Shots
    # We dynamically load real-world JSON arrays researched per-model.
    # Explicitly skip few-shots for non-Groq providers (modelslab/gemini) to
    # keep advanced CoT/ToT/ReAct prompts clean and consistent with provider guidelines.
    few_shots_block = ""
    skip_few_shots = False
    try:
        from app.infra.model_registry import PHASE_MODELS
        phase_cfg = PHASE_MODELS.get(stage, {})
        if phase_cfg.get("provider") in ("modelslab", "gemini"):
            skip_few_shots = True
    except Exception:
        pass
    # If paid providers are present, disable few-shots globally to avoid drift.
    if os.getenv("MODELSLAB_API_KEY") or os.getenv("GEMINI_API_KEY"):
        skip_few_shots = True

    model_key = (model or "").lower()
    if skip_few_shots or model_key.startswith("modelslab/") or model_key.startswith("gemini/"):
        few_shots_path = None
    else:
        few_shots_path = os.path.join(
            os.path.dirname(__file__),
            "few_shots",
            f"{stage}_{model.replace('/', '_').replace('-', '_')}.json"
        )

    if few_shots_path and os.path.exists(few_shots_path):
        try:
            with open(few_shots_path, "r", encoding="utf-8") as f:
                examples = json.load(f)
                if examples:
                    import random
                    # Dynamic Context Packing: Shuffle to prevent overfitting, limit by character budget
                    random.shuffle(examples)
                    
                    few_shot_max_chars = int(os.getenv("FEW_SHOT_MAX_CHARS", "2500"))
                    current_chars = 0
                    
                    few_shots_block = "\n\n[START FEW-SHOT EXAMPLES]\n"
                    for ex in examples:
                        if not isinstance(ex, dict):
                            continue
                            
                        ex_str = f"User: {ex.get('user', '')}\nAssistant: {ex.get('assistant_output', '')}\n---\n"
                        if current_chars + len(ex_str) > few_shot_max_chars:
                            break
                            
                        few_shots_block += ex_str
                        current_chars += len(ex_str)
                        
                    few_shots_block += "[END FEW-SHOT EXAMPLES]"
        except Exception as e:
            logger.warning(f"[PROMPT_ENGINE] Failed to parse few_shots for {stage} ({model}): {e}")

    # Compile Final Payload
    compiled = f"{persona_block}{base_template}{few_shots_block}"
    return compiled
