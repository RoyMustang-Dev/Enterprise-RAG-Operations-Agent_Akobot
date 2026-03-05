import os
import logging
import sqlite3

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv(override=True)

from app.infra.model_registry import get_phase_model
from app.infra.llm_client import run_chat_completion

class PersonaBootstrapper:
    """
    The Global Persona Initialization Controller.
    Takes simple user prompts from the frontend and expands them into 
    hyper-detailed ReAct/Tree-of-Thought System Directives using `openai/gpt-oss-120b`.
    """
    def __init__(self):
        cfg = get_phase_model("bootstrapper")
        self.provider = cfg["provider"]
        self.model_id = cfg["model"]
        self.temperature = cfg.get("temperature", 0.2)
        self.max_tokens = cfg.get("max_tokens", 1200)
        self.db_path = "data/agent_profiles.db"
        self._initialize_db()

    def _initialize_db(self):
        """Builds the strict SQLite schema mandated by the blueprint."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_name TEXT NOT NULL,
                    logo_path TEXT NOT NULL,
                    brand_details TEXT NOT NULL,
                    welcome_message TEXT NOT NULL,
                    raw_prompt TEXT,
                    expanded_prompt TEXT NOT NULL,
                    creator_user_id TEXT DEFAULT 'system',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[BOOTSTRAPPER] SQLite DB Initialization failed: {e}")

    def expand_persona(self, raw_instructions: str, bot_name: str, brand_details: str) -> str:
        """
        Pings the configured LLM synchronously to expand a 1-sentence prompt 
        into a massive ReAct framework contextually tied to the brand.
        """
        if not (os.getenv("MODELSLAB_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")):
            logger.warning("[BOOTSTRAPPER] Missing paid provider API keys. Defaulting to exact raw text.")
            return raw_instructions

        if not raw_instructions or len(raw_instructions.strip()) < 5:
            return "You are a helpful and deterministic enterprise AI assistant."

        system_instruction = f"""SYSTEM: You are an elite AI Persona Architect. 
Your job is to take the user's brief bot instructions and expand them into a highly robust, ReAct-compliant (Reason/Act) System Prompt that an underlying Enterprise LLM will follow perfectly.

Target Bot Name: {bot_name}
Target Brand Details: {brand_details}

CRITICAL RULES:
1. Ensure the final output instructs the model to use <think> tags if exploring complex logic.
2. Outline clear directives for tone, formatting (Markdown), and strict adherence to provided contexts so it doesn't hallucinate.
3. Incorporate the Brand Details implicitly.
4. Output ONLY the raw expanded prompt text. Do not output conversational padding (e.g. "Here is your expanded prompt...")."""

        try:
            logger.info(f"[BOOTSTRAPPER] Expanding core instructions via {self.provider}/{self.model_id}...")
            data = run_chat_completion(
                provider=self.provider,
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"USER RAW INSTRUCTIONS:\n{raw_instructions}"},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=30,
            )
            expanded_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not expanded_text:
                return raw_instructions
            return expanded_text
        except Exception as e:
            logger.error(f"[BOOTSTRAPPER] Expansion Engine Fault: {e}")
            return raw_instructions

    def persist_agent(self, bot_name: str, logo_path: str, brand_details: str, 
                      welcome_message: str, raw_prompt: str, expanded_prompt: str, user_id: str = "system") -> bool:
        """Saves the fully bootstrapped Persona into SQLite, forcing dynamic caches to update."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO agent_profiles (bot_name, logo_path, brand_details, welcome_message, raw_prompt, expanded_prompt, creator_user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (bot_name, logo_path, brand_details, welcome_message, raw_prompt, expanded_prompt, user_id))
            conn.commit()
            conn.close()
            
            # Immediately force the Singleton cache to pull the new Persona into RAM
            try:
                from app.prompt_engine.groq_prompts.config import PersonaCacheManager
                cache = PersonaCacheManager()
                cache.refresh_cache()
            except ImportError:
                pass
                
            return True
        except Exception as e:
            logger.error(f"[BOOTSTRAPPER] SQLite DB Insertion failed: {e}")
            return False
