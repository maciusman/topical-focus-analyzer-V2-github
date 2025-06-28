from fastapi import APIRouter, HTTPException, Depends
import logging
from typing import List, Dict, Any

from app.services import analytics_service, qdrant_service
from app.models.analysis_models import ProjectAnalyticsResults #, UrlAnalysisDetail (not used directly here yet)
from qdrant_client import QdrantClient

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{project_name}", summary="Get Project Analytics Results", response_model=ProjectAnalyticsResults)
async def get_project_analytics_results(
    project_name: str,
    qdrant_cli: QdrantClient = Depends(qdrant_service.get_qdrant_client)
):
    """
    Retrieves the calculated analytics results for a given project.
    This includes Site Focus Score, Site Radius Score, and potential cannibalization list.
    """
    logger.info(f"Request received for analytics results of project: {project_name}")

    # Sanitize project_name (although qdrant_service might also do it)
    sanitized_project_name = project_name.lower().replace(" ", "_").replace("-", "_")
    sanitized_project_name = "".join(c for c in sanitized_project_name if c.isalnum() or c == '_')

    if not sanitized_project_name:
        raise HTTPException(status_code=400, detail="Invalid project name provided.")

    try:
        # Check if collection exists first to give a clear error
        # This is a basic check; analytics_service will handle empty collections.
        qdrant_cli.get_collection(collection_name=sanitized_project_name)
    except Exception: # Catches Qdrant's specific not found or other errors
        logger.warning(f"Project collection '{sanitized_project_name}' not found in Qdrant.")
        raise HTTPException(status_code=404, detail=f"Project '{sanitized_project_name}' not found. Please ensure analysis has been run.")

    results = await analytics_service.calculate_project_analytics(sanitized_project_name, qdrant_cli)

    if not results:
        logger.error(f"Failed to calculate or retrieve analytics for project {sanitized_project_name}.")
        # Provide a more specific error based on why results might be None (e.g., no data, calculation error)
        # For now, a generic 500 or 404 if implies no data processed.
        # If calculate_project_analytics returns None due to no points, it's more like a 404 for "results".
        raise HTTPException(status_code=404, detail=f"Analytics results not available or project empty for '{sanitized_project_name}'. Ensure URLs have been processed.")

    return results


@router.get("/{project_name}/coordinates", summary="Get Dimensionality Reduced Coordinates")
async def get_project_coordinates(
    project_name: str,
    method: str = "umap", # Allow choosing method, default to umap
    n_components: int = 2, # Default to 2D
    qdrant_cli: QdrantClient = Depends(qdrant_service.get_qdrant_client)
) -> List[Dict[str, Any]]: # Return type based on analytics_service.get_dimensionality_reduced_coordinates
    """
    Retrieves dimensionality-reduced coordinates (e.g., UMAP or t-SNE)
    for all data points in a project, suitable for visualization.
    """
    logger.info(f"Request received for coordinates of project: {project_name} (method: {method}, components: {n_components})")

    sanitized_project_name = project_name.lower().replace(" ", "_").replace("-", "_")
    sanitized_project_name = "".join(c for c in sanitized_project_name if c.isalnum() or c == '_')

    if not sanitized_project_name:
        raise HTTPException(status_code=400, detail="Invalid project name provided.")
    if n_components not in [2, 3]:
        raise HTTPException(status_code=400, detail="Number of components must be 2 or 3.")
    if method not in ["umap", "tsne"]:
        raise HTTPException(status_code=400, detail="Reduction method must be 'umap' or 'tsne'.")

    try:
        # Check if collection exists
        qdrant_cli.get_collection(collection_name=sanitized_project_name)
    except Exception:
        logger.warning(f"Project collection '{sanitized_project_name}' not found for coordinates.")
        raise HTTPException(status_code=404, detail=f"Project '{sanitized_project_name}' not found.")

    try:
        coordinates = await analytics_service.get_dimensionality_reduced_coordinates(
            sanitized_project_name, qdrant_cli, method=method, n_components=n_components
        )
        if not coordinates: # Could be empty if no data or error during reduction
            # Distinguish between "no data to reduce" and "error during reduction"
            # get_dimensionality_reduced_coordinates logs errors, so here we assume it might be empty data
            logger.info(f"No coordinates generated for project {sanitized_project_name}, possibly due to no data or few points.")
            # Return empty list with 200 OK if processing was fine but no data,
            # or if an error occurred, it should have been logged by the service.
            # The service might raise an error for critical issues like missing libraries.
    except ImportError as e:
        logger.error(f"Missing dependency for dimensionality reduction: {e}")
        raise HTTPException(status_code=501, detail=f"Missing dependency for {method}: {e}. Please install required libraries.")
    except ValueError as e: # e.g. from bad method string, though validated above
        logger.error(f"ValueError during coordinate generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e: # Catch-all for other unexpected errors from the service
        logger.error(f"Unexpected error generating coordinates for {sanitized_project_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while generating coordinates.")

    return coordinates

# This router needs to be included in app/main.py
# from .api.endpoints import results
# app.include_router(results.router, prefix="/api/v1/results", tags=["Analysis Results"])
