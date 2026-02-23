"""
Agent Configuration & Master Settings Router

Provides endpoints meant for handling external dashboard logic where clients 
can inject highly custom system prompts, branding, or operational rules.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

# Initialize the modular REST router prefix
router = APIRouter(
    prefix="/api/v1/settings",
    tags=["Agent & System Config"],
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class BotConfig(BaseModel):
    """
    Pydantic Schema defining the strict configuration variables
    that dictate how the generic Chatbot represents itself and operates.
    """
    bot_name: str               # The display name of the Agent
    company_name: str           # The owning enterprise entity 
    brand_details: str          # Style guidelines (e.g., 'professional but conversational')
    welcome_message: str        # Hardcoded message shown when sessions start
    prompt_instructions: str    # Internal logic injected into the Supervisor system prompt

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@router.post("/configure")
async def update_agent_config(config: BotConfig):
    """
    **Configure the Agent Personality (Dashboard Hook)**
    
    Accepts the client's payload detailing their Bot Name, Brand Details, and Custom Prompt rules.
    *(Currently acts as a functional shell until Phase 10 Multi-Tenant storage mapping is written)*
    """
    try:
        logger.info(f"Received new Agent config for: {config.bot_name}")
        # TODO: Phase 10 Relational Implementation (SQLite / Postgres hook)
        # We will parse this dict and insert into the Tenant Configuration database.
        
        return {"status": "success", "message": "Bot configuration updated successfully."}
    except Exception as e:
        logger.error(f"Config Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to parse config.")

@router.get("/status")
async def get_agent_status():
    """
    **System Sanity Check Endpoint**
    
    Returns the current active deployment configuration and confirms API operational readiness.
    """
    return {
        "active_llm": "pending_routing", # The actual LLM is dynamically evaluated by LLMRouter
        "bot_name": "AKO AI Assistant",
        "multimodal_enabled": True
    }
