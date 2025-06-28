import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch

# Mock settings before import
from app.core import config as app_config
app_config.settings.FOCUS_K_FACTOR = 5.0
app_config.settings.RADIUS_K_FACTOR = 5.0
app_config.settings.CANNIBALIZATION_DISTANCE_THRESHOLD = 0.1

from app.services import analytics_service
from app.models.analysis_models import ProjectAnalyticsResults, CannibalizationPair
from qdrant_client import models as qdrant_models # Renamed to avoid conflict with Pydantic models

# --- Fixtures for test data ---
@pytest.fixture
def sample_vectors():
    # Simple 3D vectors for testing
    return np.array([
        [0.1, 0.2, 0.3],  # Vector 0
        [0.4, 0.5, 0.6],  # Vector 1
        [0.15, 0.25, 0.35], # Vector 2 (close to Vector 0)
        [0.8, 0.9, 1.0]   # Vector 3 (further away)
    ], dtype=np.float32)

@pytest.fixture
def sample_qdrant_points(sample_vectors):
    points = []
    for i, vec in enumerate(sample_vectors):
        points.append(qdrant_models.Record(
            id=f"point_id_{i}",
            payload={"url": f"http://example.com/page{i}", "content_md": f"Content for page {i}"},
            vector=vec.tolist() # Ensure it's a list for the Record model
        ))
    return points

@pytest.fixture
def mock_qdrant_client_for_analytics():
    mock_client = MagicMock()
    # scroll will be patched per test or configured here if consistent
    return mock_client

# --- Tests for calculate_centroid ---
def test_calculate_centroid_empty():
    assert analytics_service.calculate_centroid(np.array([])) is None

def test_calculate_centroid_single_vector():
    vector = np.array([[1.0, 0.0, 0.0]], dtype=np.float32) # Already normalized
    # Normalizing a single vector results in itself if already normalized.
    # calculate_centroid normalizes the mean.
    expected_centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    centroid = analytics_service.calculate_centroid(vector)
    assert centroid is not None
    np.testing.assert_array_almost_equal(centroid, expected_centroid, decimal=6)

def test_calculate_centroid_multiple_vectors(sample_vectors):
    # Expected: mean of vectors, then normalize the mean vector
    raw_mean = np.mean(sample_vectors, axis=0)
    expected_centroid = raw_mean / np.linalg.norm(raw_mean)

    centroid = analytics_service.calculate_centroid(sample_vectors)
    assert centroid is not None
    np.testing.assert_array_almost_equal(centroid, expected_centroid, decimal=6)


# --- Tests for calculate_project_analytics ---
@pytest.mark.asyncio
async def test_calculate_project_analytics_no_points(mock_qdrant_client_for_analytics):
    mock_qdrant_client_for_analytics.scroll = MagicMock(return_value=([], None)) # No points
    project_name = "empty_project"

    results = await analytics_service.calculate_project_analytics(project_name, mock_qdrant_client_for_analytics)

    assert results is None
    mock_qdrant_client_for_analytics.scroll.assert_called_once()

@pytest.mark.asyncio
async def test_calculate_project_analytics_no_vectors(mock_qdrant_client_for_analytics):
    # Points exist but have no vectors
    points_no_vectors = [
        qdrant_models.Record(id="pv_1", payload={"url": "url1"}, vector=None),
        qdrant_models.Record(id="pv_2", payload={"url": "url2"}, vector=None)
    ]
    mock_qdrant_client_for_analytics.scroll = MagicMock(return_value=(points_no_vectors, None))
    project_name = "no_vectors_project"

    results = await analytics_service.calculate_project_analytics(project_name, mock_qdrant_client_for_analytics)
    assert results is None # Expect None as no vectors to process

@pytest.mark.asyncio
async def test_calculate_project_analytics_single_point(mock_qdrant_client_for_analytics, sample_qdrant_points):
    single_point_list = [sample_qdrant_points[0]]
    mock_qdrant_client_for_analytics.scroll = MagicMock(return_value=(single_point_list, None))
    project_name = "single_point_project"

    results = await analytics_service.calculate_project_analytics(project_name, mock_qdrant_client_for_analytics)

    assert results is not None
    assert results.project_name == project_name
    assert results.total_urls_analyzed == 1
    # For a single point:
    # Centroid is the point itself (normalized).
    # Distance from centroid is 0.
    # Max distance from centroid is 0.
    # Max pairwise distance is 0 (or undefined, treated as 0).
    # Focus score should be 100.
    # Radius score should be 0.
    assert results.site_focus_score == pytest.approx(100.0)
    assert results.site_radius_score == pytest.approx(0.0)
    assert len(results.potential_cannibalization) == 0
    assert results.centroid_vector is not None
    np.testing.assert_array_almost_equal(
        np.array(results.centroid_vector),
        sample_qdrant_points[0].vector / np.linalg.norm(sample_qdrant_points[0].vector),
        decimal=6
    )


@pytest.mark.asyncio
async def test_calculate_project_analytics_full_run(mock_qdrant_client_for_analytics, sample_qdrant_points, sample_vectors):
    mock_qdrant_client_for_analytics.scroll = MagicMock(return_value=(sample_qdrant_points, None))
    project_name = "full_analytics_project"

    results = await analytics_service.calculate_project_analytics(project_name, mock_qdrant_client_for_analytics)

    assert results is not None
    assert results.project_name == project_name
    assert results.total_urls_analyzed == len(sample_vectors)

    # Focus and Radius scores depend on the k1, k2 factors and distances.
    # We won't replicate the exact math here but check they are within [0, 100].
    assert 0 <= results.site_focus_score <= 100
    assert 0 <= results.site_radius_score <= 100

    # Cannibalization: Vector 0 and Vector 2 are close.
    # sample_vectors[0] = [0.1, 0.2, 0.3]
    # sample_vectors[2] = [0.15, 0.25, 0.35]
    # Cosine distance between them:
    from sklearn.metrics.pairwise import cosine_distances
    dist_0_2 = cosine_distances(sample_vectors[0].reshape(1, -1), sample_vectors[2].reshape(1, -1))[0,0]

    # app_config.settings.CANNIBALIZATION_DISTANCE_THRESHOLD = 0.1
    # Check if dist_0_2 is less than this threshold
    found_cannibalization = False
    for pair in results.potential_cannibalization:
        if (pair.url1 == "http://example.com/page0" and pair.url2 == "http://example.com/page2") or \
           (pair.url1 == "http://example.com/page2" and pair.url2 == "http://example.com/page0"):
            found_cannibalization = True
            assert pair.similarity_score == pytest.approx(1.0 - dist_0_2, abs=1e-6)
            break

    if dist_0_2 < app_config.settings.CANNIBALIZATION_DISTANCE_THRESHOLD:
        assert found_cannibalization, "Expected cannibalization between page0 and page2 not found."
        assert len(results.potential_cannibalization) >= 1
    else:
        assert not found_cannibalization, "Unexpected cannibalization found, distance might be above threshold."

    assert results.centroid_vector is not None
    # Centroid calculation was tested separately.

# --- Tests for get_dimensionality_reduced_coordinates ---
# These tests would require mocking umap.UMAP and TSNE, or allowing them to run on small data.
# For now, let's test the service's ability to call these and handle data.

@pytest.mark.asyncio
@patch('app.services.analytics_service.umap.UMAP') # Patch UMAP at the location it's imported
async def test_get_coordinates_umap_success(mock_umap_constructor, mock_qdrant_client_for_analytics, sample_qdrant_points):
    # Mock UMAP instance and its fit_transform method
    mock_umap_instance = MagicMock()
    # Create dummy coordinates that match the number of sample points and n_components
    # sample_qdrant_points has 4 points. n_components=2 by default.
    dummy_coords = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    mock_umap_instance.fit_transform.return_value = dummy_coords
    mock_umap_constructor.return_value = mock_umap_instance

    mock_qdrant_client_for_analytics.scroll = MagicMock(return_value=(sample_qdrant_points, None))
    project_name = "umap_project"

    coordinates_result = await analytics_service.get_dimensionality_reduced_coordinates(
        project_name, mock_qdrant_client_for_analytics, method="umap", n_components=2
    )

    assert len(coordinates_result) == len(sample_qdrant_points)
    mock_umap_constructor.assert_called_once_with(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, metric='cosine')
    mock_umap_instance.fit_transform.assert_called_once() # Check it was called with the vectors

    # Check input to fit_transform
    call_args = mock_umap_instance.fit_transform.call_args[0][0]
    expected_vectors = np.array([p.vector for p in sample_qdrant_points])
    np.testing.assert_array_almost_equal(call_args, expected_vectors, decimal=6)

    for i, res_item in enumerate(coordinates_result):
        assert res_item["id"] == sample_qdrant_points[i].id
        assert res_item["url"] == sample_qdrant_points[i].payload["url"]
        assert res_item["x"] == pytest.approx(dummy_coords[i, 0])
        assert res_item["y"] == pytest.approx(dummy_coords[i, 1])


@pytest.mark.asyncio
@patch('app.services.analytics_service.TSNE') # Patch TSNE
async def test_get_coordinates_tsne_success(mock_tsne_constructor, mock_qdrant_client_for_analytics, sample_qdrant_points):
    mock_tsne_instance = MagicMock()
    dummy_coords = np.array([[1.1, 1.2], [1.3, 1.4], [1.5, 1.6], [1.7, 1.8]], dtype=np.float32)
    mock_tsne_instance.fit_transform.return_value = dummy_coords
    mock_tsne_constructor.return_value = mock_tsne_instance

    mock_qdrant_client_for_analytics.scroll = MagicMock(return_value=(sample_qdrant_points, None))
    project_name = "tsne_project"

    coordinates_result = await analytics_service.get_dimensionality_reduced_coordinates(
        project_name, mock_qdrant_client_for_analytics, method="tsne", n_components=2
    )

    assert len(coordinates_result) == len(sample_qdrant_points)
    # Perplexity for 4 samples is min(30, 4-1) = 3
    mock_tsne_constructor.assert_called_once_with(n_components=2, random_state=42, perplexity=3, metric='cosine', n_jobs=-1)
    mock_tsne_instance.fit_transform.assert_called_once()


@pytest.mark.asyncio
async def test_get_coordinates_no_points(mock_qdrant_client_for_analytics):
    mock_qdrant_client_for_analytics.scroll = MagicMock(return_value=([], None))
    project_name = "no_points_for_umap"

    coordinates_result = await analytics_service.get_dimensionality_reduced_coordinates(
        project_name, mock_qdrant_client_for_analytics, method="umap"
    )
    assert coordinates_result == []

@pytest.mark.asyncio
async def test_get_coordinates_import_error(mock_qdrant_client_for_analytics, sample_qdrant_points):
    mock_qdrant_client_for_analytics.scroll = MagicMock(return_value=(sample_qdrant_points, None))
    project_name = "import_error_project"

    # Simulate ImportError when 'umap' is accessed
    with patch('app.services.analytics_service.umap', new=None): # Or raise ImportError directly
        with pytest.raises(ImportError): # analytics_service should re-raise it
             await analytics_service.get_dimensionality_reduced_coordinates(
                project_name, mock_qdrant_client_for_analytics, method="umap"
            )

# To run: poetry run pytest tests/test_analytics_service.py
# Ensure umap-learn and scikit-learn are in dev dependencies if not main, for tests to pass if not mocking.
# For these tests, umap and tsne are mocked, so the libraries aren't strictly needed at test runtime *if* mocks are perfect.
# However, it's good practice for them to be available in the test environment.
# The current pyproject.toml includes them in main dependencies.
