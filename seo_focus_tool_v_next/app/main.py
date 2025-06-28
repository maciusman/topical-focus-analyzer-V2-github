from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="SEO Focus Tool API",
    version="0.1.0",
    description="API for analyzing Site Focus Score and Site Radius Score using Jina AI, Qdrant, and OpenRouter."
)

# Import routers
from app.api.endpoints import analysis as analysis_router
from app.api.endpoints import results as results_router
from app.api.endpoints import llm as llm_router # Created in Module 5

@app.get("/health", summary="Health Check", tags=["General"])
async def health_check():
    """
    Endpoint to check the health of the API.
    Returns a simple status message.
    """
    return {"status": "healthy", "message": "API is running!"}

# Include routers
app.include_router(analysis_router.router, prefix="/api/v1/analysis", tags=["Analysis Processing"])
app.include_router(results_router.router, prefix="/api/v1/results", tags=["Analysis Results"])
app.include_router(llm_router.router, prefix="/api/v1/llm", tags=["LLM Service"])


if __name__ == "__main__":
    import uvicorn
    # This is for local development running this file directly
    # For production, use a proper ASGI server like Uvicorn or Hypercorn, e.g.:
    # uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
