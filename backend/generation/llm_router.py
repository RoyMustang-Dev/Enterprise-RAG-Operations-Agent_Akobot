"""
Dynamic LLM Routing Multiplexer

This module abstracts the exact LLM Engine initialization from the agent scripts.
It enforces the Architecture pattern: "Sarvam for Core Reasoning, Gemini purely for Situational Vision".
"""
import os
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class LLMRouter:
    """
    Dynamically routes LLM instantiations based on dynamic tool requirements.
    Acts as the single point of truth mapping keys to their specific Python Class Wrappers.
    """
    
    @staticmethod
    def get_best_available_llm(temperature: float = 0.2, target_provider: Optional[str] = None, api_key: Optional[str] = None) -> Any:
        """
        Initializes the optimal Language Model client dynamically based on available API Keys
        and requested capability (e.g., Vision).
        
        Args:
            temperature (float): Baseline default temperature passed down.
            target_provider (str, optional): Hard override. e.g., 'gemini' for Multimodal Images.
            api_key (str, optional): Target key for Sarvam.
            
        Returns:
            Any: An instantiated client (SarvamClient || ChatGoogleGenerativeAI).
        """
        # Determine strict provider target, default to Sarvam Enterprise
        provider = (target_provider or "sarvam").lower()
        logger.info(f"LLM Router: Initializing provider constraint: {provider}")

        # -------------------------------------------------------------------------
        # 1. Situational Vision Fallback (Gemini Node)
        # -------------------------------------------------------------------------
        # If the Chat API explicitly recognized `image_file` blobs, it forces this router
        # to fetch Gemini 1.5 Flash solely to extract text, skipping Sarvam temporarily.
        if provider == 'gemini':
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=temperature)
                except ImportError:
                    logger.error("langchain-google-genai not installed. Cannot initialize Vision Node.")
            logger.error("Gemini targeted by Supervisor but key/library is missing. Aborting payload.")
            return None
                    
        # -------------------------------------------------------------------------
        # 2. Core Operational Generation Provider (Sarvam Node)
        # -------------------------------------------------------------------------
        elif provider == 'sarvam':
            from dotenv import load_dotenv
            load_dotenv()
            # Failsafe sequence to securely read environment vs ephemeral payload parameter keys
            final_key = api_key or os.getenv("SARVAM_API_KEY")
            if final_key:
                try:
                    from backend.generation.llm_provider import SarvamClient
                    logger.info("Initializing Sarvam Enterprise Architecture Module...")
                    return SarvamClient(api_key=final_key)
                except Exception as e:
                    logger.error(f"Sarvam API generic instantiation failed: {e}")
            else:
                logger.error("SARVAM_API_KEY is missing from environment variables bounds.")
                raise ValueError("SARVAM_API_KEY is strictly required but not found in .env or system environment.")
                
        return None
