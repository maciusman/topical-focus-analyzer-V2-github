import os
from dotenv import load_dotenv
from typing import Optional

# Load .env file variables
load_dotenv()

class Settings:
    PROJECT_NAME: str = "SEO Focus Tool API"
    VERSION: str = "0.1.0"

    # API Keys
    JINA_API_KEY: Optional[str] = os.getenv("JINA_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")

    # Qdrant settings
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", "6334")) # Default gRPC port for Qdrant
    # QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") # If Qdrant is secured

    # Embedding model details (consistent with Jina v4)
    EMBEDDING_MODEL_DIMENSION: int = 2048 # For jina-embeddings-v4
    EMBEDDING_MODEL_DISTANCE_METRIC: str = "Cosine" # Qdrant's naming for Distance.COSINE

    # Default LLM model for OpenRouter (can be overridden by user)
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "openai/gpt-3.5-turbo")

    # Base URL for FastAPI (used by Streamlit to connect)
    FASTAPI_BASE_URL: str = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")


settings = Settings()

# Validate essential API keys on import
# if not settings.JINA_API_KEY:
#     raise ValueError("JINA_API_KEY is not set in the environment variables.")
# if not settings.OPENROUTER_API_KEY:
#     raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")

# Note: For a real application, consider more robust validation or logging for missing keys,
# especially if some functionalities can work without certain keys.
# For now, warnings might be printed by services that use them if they are missing.
