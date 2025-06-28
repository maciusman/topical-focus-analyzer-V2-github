from fastapi import APIRouter, HTTPException, Depends, Body
import logging
from typing import List, Optional

from app.services import llm_service, qdrant_service, analytics_service
from app.models.analysis_models import LlmModel, ProjectAnalyticsResults
from qdrant_client import QdrantClient

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/models", summary="Get Available LLM Models from OpenRouter", response_model=List[LlmModel])
async def get_available_llm_models():
    """
    Retrieves a list of available LLM models from OpenRouter.
    The list might be filtered or curated by the `llm_service`.
    """
    logger.info("Request received for available LLM models.")
    models = await llm_service.get_openrouter_models()
    if not models:
        # This could be due to API key issue or OpenRouter service issue.
        # llm_service logs details.
        raise HTTPException(status_code=503, detail="Could not retrieve models from OpenRouter at this time.")
    return models

@router.post("/{project_name}/summarize", summary="Generate AI Summary for Project Analysis")
async def generate_project_summary(
    project_name: str,
    model_name: str = Body(..., description="The ID of the OpenRouter model to use for summarization."),
    qdrant_cli: QdrantClient = Depends(qdrant_service.get_qdrant_client)
):
    """
    Generates an AI-powered summary for the specified project's analysis results
    using the selected OpenRouter LLM.
    """
    logger.info(f"Request received to generate summary for project: {project_name} using model: {model_name}")

    sanitized_project_name = project_name.lower().replace(" ", "_").replace("-", "_")
    sanitized_project_name = "".join(c for c in sanitized_project_name if c.isalnum() or c == '_')

    if not sanitized_project_name:
        raise HTTPException(status_code=400, detail="Invalid project name provided.")

    # 1. Fetch analysis results for the project
    # These results will form the context for the LLM.
    analysis_results = await analytics_service.calculate_project_analytics(sanitized_project_name, qdrant_cli)
    if not analysis_results:
        raise HTTPException(status_code=404, detail=f"Analysis results not found for project '{sanitized_project_name}'. Cannot generate summary.")

    # 1.b. Fetch top focused/divergent URLs (if not part of ProjectAnalyticsResults directly)
    # This part needs enhancement: analytics_service needs to provide these lists,
    # or this endpoint needs to compute them based on all URL data + centroid.
    # For now, passing None, the LLM prompt will indicate N/A for these.
    # TODO: Implement logic to fetch/calculate top_focused_urls and top_divergent_urls.
    # This would likely involve:
    # - Getting all points with vectors and payloads.
    # - Calculating centroid (if not already in analysis_results.centroid_vector).
    # - Calculating distances for all points from centroid.
    # - Sorting and picking top/bottom N.
    # This is non-trivial and might be slow for large projects if done on-the-fly here.
    # A better approach is for `calculate_project_analytics` to optionally return this.
    top_focused_urls: Optional[List[str]] = None
    top_divergent_urls: Optional[List[str]] = None
    # Example (conceptual - needs full data from Qdrant & centroid):
    # if analysis_results.centroid_vector:
    #     all_url_data_for_sorting = await some_service.get_all_url_details_with_vectors(sanitized_project_name, qdrant_cli)
    #     # ... calculate distances, sort, get top N ...
    #     # top_focused_urls = ...
    #     # top_divergent_urls = ...


    # 2. Generate summary using the LLM service
    summary_text = await llm_service.generate_summary_openrouter(
        project_name=sanitized_project_name,
        model_name=model_name,
        analysis_results=analysis_results,
        top_focused_urls=top_focused_urls, # Pass the fetched/calculated lists
        top_divergent_urls=top_divergent_urls
    )

    if not summary_text:
        # llm_service logs details of the error.
        raise HTTPException(status_code=500, detail=f"Failed to generate summary for project '{sanitized_project_name}' using model '{model_name}'.")

    return {
        "project_name": sanitized_project_name,
        "model_used": model_name,
        "summary": summary_text
    }

# This router needs to be included in app/main.py
# from .api.endpoints import llm
# app.include_router(llm.router, prefix="/api/v1/llm", tags=["LLM Service"])
