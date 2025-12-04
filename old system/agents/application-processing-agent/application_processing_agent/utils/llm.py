import logging
from langchain_openai import ChatOpenAI
try:
    from .settings import agent_settings
except ImportError:
    from settings import agent_settings

logger = logging.getLogger(__name__)

def get_llm():
    """Initialize and return the LLM based on the configured provider."""
    if not agent_settings.AP_LLM_API_KEY:
        raise ValueError("AP_LLM_API_KEY is required but not set")
    
    if not agent_settings.AP_LLM_ENDPOINT:
        raise ValueError("AP_LLM_ENDPOINT is required but not set")
    
    if not agent_settings.AP_LLM_MODEL:
        raise ValueError("AP_LLM_MODEL is required but not set")
    
    try:    
        logger.info(f"Initializing LiteLLM model: {agent_settings.AP_LLM_MODEL}")
        logger.info(f"Using endpoint: {agent_settings.AP_LLM_ENDPOINT}")
        
        return ChatOpenAI(
            model=agent_settings.AP_LLM_MODEL,
            openai_api_base=agent_settings.AP_LLM_ENDPOINT,
            openai_api_key=agent_settings.AP_LLM_API_KEY,
            temperature=0,
            streaming=False
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        raise ValueError(f"Failed to initialize LLM. Check configuration: {e}")

def check_llm_env_vars():    
    """Validate LLM environment variables and test connection."""
    # Check required environment variables
    required_vars = {
        'AP_LLM_API_KEY': agent_settings.AP_LLM_API_KEY,
        'AP_LLM_ENDPOINT': agent_settings.AP_LLM_ENDPOINT,
        'AP_LLM_MODEL': agent_settings.AP_LLM_MODEL
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Try to initialize to catch any other connection errors
    try:
        get_llm()
        logger.info("Successfully initialized LLM connection")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        raise ValueError(f"Failed to initialize LLM. Check credentials/endpoint: {e}")