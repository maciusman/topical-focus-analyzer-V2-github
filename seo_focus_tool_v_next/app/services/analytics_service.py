import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any
from sklearn.metrics.pairwise import cosine_distances
# cosine_distances = 1 - cosine_similarity. For embeddings, similarity is natural, distance is 1-sim.
from sklearn.preprocessing import normalize

from app.services import qdrant_service
from app.models.analysis_models import ProjectAnalyticsResults, CannibalizationPair, UrlAnalysisDetail
from app.core.config import settings

# Dimensionality reduction tools will be added here if UMAP/t-SNE is done on backend
# from sklearn.manifold import TSNE
# import umap

logger = logging.getLogger(__name__)

# Epsilon for numerical stability
epsilon = 1e-9

def calculate_centroid(vectors: np.ndarray) -> Optional[np.ndarray]:
    """Calculates the centroid (mean vector) of a list of vectors."""
    if vectors.size == 0:
        return None
    # Normalize each vector before averaging for cosine distance space
    # normalized_vectors = normalize(vectors, axis=1, norm='l2')
    # centroid = np.mean(normalized_vectors, axis=0)
    # It's often better to average directly and then normalize the centroid,
    # or work with sums for spherical k-means type centroids.
    # For simple average:
    centroid = np.mean(vectors, axis=0)
    return normalize(centroid.reshape(1, -1), axis=1, norm='l2')[0]


async def calculate_project_analytics(project_name: str, qdrant_cli: qdrant_service.QdrantClient) -> Optional[ProjectAnalyticsResults]:
    """
    Calculates analytics for a given project:
    - Site Focus Score
    - Site Radius Score
    - Cannibalization
    - Data for visualizations (coordinates - placeholder for now)
    """
    sanitized_project_name = project_name.lower().replace(" ", "_").replace("-", "_")
    sanitized_project_name = "".join(c for c in sanitized_project_name if c.isalnum() or c == '_')

    try:
        # 1. Retrieve all points (vectors and payloads) from Qdrant for the project
        # Use scroll API to get all points.
        # This might be memory intensive for very large projects.
        # Consider sampling or iterative approaches for huge datasets if memory becomes an issue.

        # Initial scroll request
        # For vector retrieval, ensure with_vectors=True
        response = qdrant_cli.scroll(
            collection_name=sanitized_project_name,
            limit=250,  # Adjust batch size as needed
            with_payload=True,
            with_vectors=True
        )
        all_points = response[0] # list of Record objects
        next_offset = response[1] # next offset_id for pagination

        while next_offset is not None:
            response = qdrant_cli.scroll(
                collection_name=sanitized_project_name,
                limit=250,
                offset=next_offset,
                with_payload=True,
                with_vectors=True
            )
            all_points.extend(response[0])
            next_offset = response[1]

        if not all_points:
            logger.warning(f"No points found in Qdrant for project {sanitized_project_name}. Cannot calculate analytics.")
            return None

        logger.info(f"Retrieved {len(all_points)} points from Qdrant for project {sanitized_project_name} analytics.")

        vectors = np.array([point.vector for point in all_points if point.vector is not None])
        if vectors.shape[0] == 0:
            logger.warning(f"No vectors found among points for project {sanitized_project_name}.")
            return None

        num_points = vectors.shape[0]

        # --- Site Focus Score & Site Radius Score ---
        site_focus_score = 0.0
        site_radius_score = 0.0

        # Calculate centroid
        centroid_vector = calculate_centroid(vectors)
        if centroid_vector is None:
            logger.warning(f"Could not calculate centroid for project {sanitized_project_name}.")
            # Cannot calculate scores without centroid
            return ProjectAnalyticsResults(
                project_name=sanitized_project_name,
                total_urls_analyzed=num_points,
                site_focus_score=0.0, # Default or error indicator
                site_radius_score=0.0,
                potential_cannibalization=[] # Other fields might be empty or default
            )

        # Calculate distances from centroid (cosine distance)
        # cosine_distances returns distances (0 to 2, or 0 to 1 for normalized vectors)
        # For normalized vectors and normalized centroid, cosine_distance = 1 - cosine_similarity
        distances_from_centroid = cosine_distances(vectors, centroid_vector.reshape(1, -1)).flatten()

        avg_distance_from_centroid = np.mean(distances_from_centroid)
        max_distance_from_centroid = np.max(distances_from_centroid)

        # Pairwise distances between all vectors
        # This can be very memory intensive: (N*N) matrix.
        # For N=1000, it's 1M floats. For N=10000, it's 100M floats (e.g. ~400MB-800MB)
        # Consider if this is truly needed for Radius score or if an approximation can be used.
        # The original app used it.
        max_pairwise_dist = 0.0
        if num_points > 1:
            pairwise_dist_matrix = cosine_distances(vectors)
            # We need the max distance, ignoring diagonal (self-distance which is 0)
            # Max of upper triangle is sufficient as matrix is symmetric.
            if pairwise_dist_matrix.size > 1:
                 # Ensure we don't include the diagonal by setting it to a value that won't be max
                 np.fill_diagonal(pairwise_dist_matrix, -1.0) # Or some other indicator
                 max_pairwise_dist = np.max(pairwise_dist_matrix[pairwise_dist_matrix >= 0])


        # Scaling factors (k1, k2) - these should ideally be configurable per project or globally
        k1 = settings.FOCUS_K_FACTOR if hasattr(settings, 'FOCUS_K_FACTOR') else 5.0
        k2 = settings.RADIUS_K_FACTOR if hasattr(settings, 'RADIUS_K_FACTOR') else 5.0

        # Focus Score Calculation (adapted from original tool)
        # Original: 100.0 * (1.0 - normalized_avg_dist)**(k1 / 5.0)
        # normalized_avg_dist = avg_distance / (max_distance_from_centroid + epsilon)
        # Cosine distance is already between 0 and 1 (for normalized vectors).
        if max_distance_from_centroid < epsilon:
            site_focus_score = 100.0
        else:
            normalized_avg_dist = avg_distance_from_centroid / (max_distance_from_centroid + epsilon)
            # Clamp normalized_avg_dist to be between 0 and 1 before power
            normalized_avg_dist = max(0.0, min(1.0, normalized_avg_dist))
            site_focus_score = 100.0 * ((1.0 - normalized_avg_dist) ** (k1 / 5.0))

        site_focus_score = max(0.0, min(100.0, site_focus_score))

        # Radius Score Calculation (adapted from original tool)
        # Original: 100.0 * (1.0 - np.exp(-radius_ratio * (k2 / 5.0)))
        # radius_ratio = max_distance_from_centroid / (max_pairwise_dist + epsilon)
        if max_pairwise_dist < epsilon:
            site_radius_score = 0.0 # Or 100.0 if all points are same, implies very small radius relative to itself
                                   # If max_pairwise_dist is 0, all points are identical.
                                   # If max_dist_from_centroid is also 0, radius_ratio is ill-defined.
                                   # If all points are identical, radius should be minimal.
            site_radius_score = 0.0 if max_distance_from_centroid < epsilon else 100.0 # If centroid is one of the points vs not
        else:
            radius_ratio = max_distance_from_centroid / (max_pairwise_dist + epsilon)
            radius_ratio = max(0.0, min(1.0, radius_ratio)) # Clamp ratio
            # The formula (1 - exp(-x)) maps x=[0, inf) to [0, 1).
            # Higher radius_ratio or k2 means higher score.
            site_radius_score = 100.0 * (1.0 - np.exp(-radius_ratio * (k2 / 5.0)))

        site_radius_score = max(0.0, min(100.0, site_radius_score))


        # --- Cannibalization Detection ---
        # For each vector, find K nearest neighbors using Qdrant's search.
        # This is more efficient than computing full pairwise matrix for this specific task if K is small.
        # However, we already computed pairwise_dist_matrix for Radius Score.
        # If pairwise_dist_matrix is available and not too large, we can reuse it.
        # For now, let's assume we might use Qdrant search for scalability if pairwise is too big.
        # For simplicity, if pairwise_dist_matrix was computed, use it.

        potential_cannibalization: List[CannibalizationPair] = []
        # cannibalization_threshold = settings.CANNIBALIZATION_THRESHOLD if hasattr(settings, 'CANNIBALIZATION_THRESHOLD') else 0.95 # Cosine SIMILARITY threshold
        # For cosine DISTANCE, threshold would be low, e.g., 1 - 0.95 = 0.05
        cannibalization_distance_threshold = settings.CANNIBALIZATION_DISTANCE_THRESHOLD if hasattr(settings, 'CANNIBALIZATION_DISTANCE_THRESHOLD') else 0.10

        if num_points > 1 and 'pairwise_dist_matrix' in locals(): # Check if matrix was computed
            # Iterate over upper triangle of the distance matrix
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    dist = pairwise_dist_matrix[i, j]
                    if dist < cannibalization_distance_threshold:
                        url1 = all_points[i].payload.get("url", f"PointID_{all_points[i].id}")
                        url2 = all_points[j].payload.get("url", f"PointID_{all_points[j].id}")
                        similarity_score = 1.0 - dist # Convert distance back to similarity for reporting
                        potential_cannibalization.append(
                            CannibalizationPair(url1=url1, url2=url2, similarity_score=similarity_score)
                        )
            # Sort by similarity (highest first)
            potential_cannibalization.sort(key=lambda x: x.similarity_score, reverse=True)


        # --- Prepare URL Details (placeholder for UMAP/t-SNE coords) ---
        # url_details_list: List[UrlAnalysisDetail] = []
        # In a full app, UMAP/t-SNE would run here on `vectors`
        # For now, fill with available data. Coordinates will be None.
        # for i, point_obj in enumerate(all_points):
        #     url_details_list.append(
        #         UrlAnalysisDetail(
        #             url=point_obj.payload.get("url", f"PointID_{point_obj.id}"),
        #             vector_id=str(point_obj.id), # Ensure ID is string
        #             distance_from_centroid=distances_from_centroid[i] if distances_from_centroid is not None else None,
        #             content_preview=point_obj.payload.get("content_md", "")[:200] + "..." if point_obj.payload.get("content_md") else ""
        #             # x_coord, y_coord, z_coord will be filled after dimensionality reduction
        #         )
        #     )

        # Store all_points data (id, payload, and calculated distance) for later UMAP and detailed results
        # This avoids recalculating or passing huge lists of UrlAnalysisDetail around.
        # The UMAP step can enrich this data with coordinates.

        # For now, the main results payload will not include the full url_details list to keep it light.
        # A separate endpoint will provide paginated/filtered URL details including UMAP coordinates.

        return ProjectAnalyticsResults(
            project_name=sanitized_project_name,
            site_focus_score=site_focus_score,
            site_radius_score=site_radius_score,
            total_urls_analyzed=num_points,
            centroid_vector=centroid_vector.tolist() if centroid_vector is not None else None,
            potential_cannibalization=potential_cannibalization[:50] # Limit to top 50 for now
            # url_details will be served by another endpoint or after UMAP
        )

    except Exception as e:
        logger.error(f"Error calculating analytics for project {sanitized_project_name}: {e}", exc_info=True)
        return None


# --- Placeholder for Dimensionality Reduction (UMAP/t-SNE) ---
# This would be a separate function called by an endpoint, possibly on demand,
# as it can be computationally intensive. It would take vectors from Qdrant.
async def get_dimensionality_reduced_coordinates(project_name: str, qdrant_cli: qdrant_service.QdrantClient, method: str = "umap", n_components: int = 2) -> List[Dict[str, Any]]:
    """
    Retrieves vectors for a project and performs dimensionality reduction.
    Returns a list of dictionaries with id, url, and coordinates.
    """
    sanitized_project_name = project_name.lower().replace(" ", "_").replace("-", "_")
    sanitized_project_name = "".join(c for c in sanitized_project_name if c.isalnum() or c == '_')

    logger.info(f"Starting dimensionality reduction ({method}, {n_components}D) for project {sanitized_project_name}")

    try:
        response = qdrant_cli.scroll(collection_name=sanitized_project_name, limit=10000, with_payload=True, with_vectors=True) # Adjust limit as needed
        all_points = response[0]
        # Add pagination if necessary for very large collections

        if not all_points:
            logger.warning(f"No points for UMAP in project {sanitized_project_name}.")
            return []

        vectors = np.array([p.vector for p in all_points if p.vector is not None])
        if vectors.shape[0] < 2 : # UMAP/TSNE need at least 2 points
            logger.warning(f"Not enough vectors ({vectors.shape[0]}) for UMAP in project {sanitized_project_name}.")
            return []

        if method == "umap":
            import umap # Local import
            reducer = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=42, metric='cosine')
        elif method == "tsne":
            from sklearn.manifold import TSNE # Local import
            # Perplexity must be less than n_samples
            perplexity_val = min(30, vectors.shape[0] -1)
            if perplexity_val <=0: perplexity_val = 1 # handle case of 1 sample if it slips through
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity_val, metric='cosine', n_jobs=-1)
        else:
            raise ValueError("Unsupported reduction method. Choose 'umap' or 'tsne'.")

        logger.info(f"Running {method} on {vectors.shape[0]} vectors...")
        coordinates = reducer.fit_transform(vectors)
        logger.info(f"{method} complete.")

        results = []
        for i, point_obj in enumerate(all_points):
            res = {
                "id": str(point_obj.id),
                "url": point_obj.payload.get("url", f"PointID_{point_obj.id}"),
                "x": float(coordinates[i, 0]) # Ensure float for JSON
            }
            if n_components > 1:
                res["y"] = float(coordinates[i, 1])
            if n_components > 2:
                res["z"] = float(coordinates[i, 2])
            results.append(res)

        return results

    except ImportError:
        logger.error(f"{method} library is not installed. Please install 'umap-learn' or 'scikit-learn'.")
        raise
    except Exception as e:
        logger.error(f"Error during dimensionality reduction for {sanitized_project_name}: {e}", exc_info=True)
        return [] # Return empty on error


if __name__ == '__main__':
    # Basic test for analytics_service
    # Requires a running Qdrant instance and a collection with some data
    logging.basicConfig(level=logging.INFO)

    async def test_analytics_runner():
        try:
            q_client = qdrant_service.get_qdrant_client()
            q_client.list_collections() # Test connection
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}. Aborting test.")
            return

        # Assume 'test_processing_worker_project' was created and populated by processing_worker test
        test_project_name = "test_processing_worker_project"

        # Check if collection exists
        collections = await qdrant_service.list_collections()
        if test_project_name not in collections:
            logger.warning(f"Test collection '{test_project_name}' not found. Populate it first (e.g., by running processing_worker.py if __name__ == '__main__').")
            # You might want to create a dummy collection with a few points here for a self-contained test
            # For now, it relies on processing_worker.py's test to populate.
            return

        logger.info(f"--- Calculating Analytics for project: {test_project_name} ---")
        analytics_results = await calculate_project_analytics(test_project_name, q_client)

        if analytics_results:
            logger.info(f"Site Focus Score: {analytics_results.site_focus_score}")
            logger.info(f"Site Radius Score: {analytics_results.site_radius_score}")
            logger.info(f"Total URLs Analyzed: {analytics_results.total_urls_analyzed}")
            logger.info(f"Potential Cannibalization ({len(analytics_results.potential_cannibalization)}):")
            for item in analytics_results.potential_cannibalization[:3]:
                logger.info(f"  - {item.url1} vs {item.url2} (Similarity: {item.similarity_score:.4f})")
        else:
            logger.error("Failed to calculate analytics.")

        logger.info(f"--- Getting UMAP Coordinates for project: {test_project_name} ---")
        coordinates = await get_dimensionality_reduced_coordinates(test_project_name, q_client, method="umap", n_components=2)
        if coordinates:
            logger.info(f"Generated {len(coordinates)} UMAP coordinates. First 3:")
            for c_data in coordinates[:3]:
                logger.info(f"  ID: {c_data['id']}, URL: {c_data['url']}, X: {c_data['x']:.3f}, Y: {c_data['y']:.3f}")
        else:
            logger.warning("Failed to generate UMAP coordinates or no data.")

    asyncio.run(test_analytics_runner())
