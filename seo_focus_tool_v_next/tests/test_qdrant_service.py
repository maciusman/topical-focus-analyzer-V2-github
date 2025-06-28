import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

# Mock settings before importing the service that uses it
# This is important if settings are used at module import time in qdrant_service
from app.core import config as app_config
app_config.settings.QDRANT_HOST = "testhost"
app_config.settings.QDRANT_PORT = 1234
app_config.settings.EMBEDDING_MODEL_DIMENSION = 10 # Smaller for tests
app_config.settings.EMBEDDING_MODEL_DISTANCE_METRIC = "Cosine"

# Now import the service
from app.services import qdrant_service
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

@pytest.fixture
def mock_qdrant_client_instance():
    mock_client = MagicMock(spec=qdrant_service.QdrantClient)
    mock_client.get_collection = MagicMock()
    mock_client.create_collection = MagicMock()
    mock_client.create_payload_index = MagicMock()
    mock_client.list_collections = AsyncMock() # If list_collections becomes async
                                            # or MagicMock if it's sync in your version
    # If qdrant_client is accessed directly as a module global, patch it there
    # If it's instantiated inside functions, patch the constructor
    return mock_client

@pytest.mark.asyncio
@patch('app.services.qdrant_service.qdrant_client', new_callable=MagicMock) # Patch the global client
async def test_get_or_create_collection_exists(mock_global_qdrant_client):
    mock_global_qdrant_client.get_collection.return_value = True # Simulate collection exists
    collection_name = "test_collection_exists"

    result = await qdrant_service.get_or_create_collection(collection_name)

    assert result is True
    sanitized_name = "test_collection_exists" # Assuming sanitization doesn't change it much
    mock_global_qdrant_client.get_collection.assert_called_once_with(collection_name=sanitized_name)
    mock_global_qdrant_client.create_collection.assert_not_called()

@pytest.mark.asyncio
@patch('app.services.qdrant_service.qdrant_client', new_callable=MagicMock)
async def test_get_or_create_collection_creates_new(mock_global_qdrant_client):
    # Simulate collection not found, then successful creation
    mock_global_qdrant_client.get_collection.side_effect = UnexpectedResponse(
        status_code=404, content=b"Not found", response=MagicMock()
    )
    mock_global_qdrant_client.create_collection.return_value = True # Simulate successful creation
    mock_global_qdrant_client.create_payload_index.return_value = True

    collection_name = "new_test_collection"
    sanitized_name = "new_test_collection" # Assuming simple name

    result = await qdrant_service.get_or_create_collection(collection_name)

    assert result is True
    mock_global_qdrant_client.get_collection.assert_called_once_with(collection_name=sanitized_name)
    mock_global_qdrant_client.create_collection.assert_called_once_with(
        collection_name=sanitized_name,
        vectors_config=models.VectorParams(
            size=app_config.settings.EMBEDDING_MODEL_DIMENSION, # Use mocked setting
            distance=models.Distance[app_config.settings.EMBEDDING_MODEL_DISTANCE_METRIC.upper()]
        )
    )
    assert mock_global_qdrant_client.create_payload_index.call_count == 2 # For 'url' and 'processed_at'
    mock_global_qdrant_client.create_payload_index.assert_any_call(
        collection_name=sanitized_name,
        field_name="url",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    mock_global_qdrant_client.create_payload_index.assert_any_call(
        collection_name=sanitized_name,
        field_name="processed_at",
        field_schema=models.PayloadSchemaType.DATETIME
    )


@pytest.mark.asyncio
@patch('app.services.qdrant_service.qdrant_client', new_callable=MagicMock)
async def test_get_or_create_collection_sanitizes_name(mock_global_qdrant_client):
    mock_global_qdrant_client.get_collection.side_effect = UnexpectedResponse(status_code=404, content=b"Not found", response=MagicMock())
    collection_name = "My Test Project-123"
    sanitized_name = "my_test_project_123"

    await qdrant_service.get_or_create_collection(collection_name)

    mock_global_qdrant_client.get_collection.assert_called_once_with(collection_name=sanitized_name)
    mock_global_qdrant_client.create_collection.assert_called_once_with(
        collection_name=sanitized_name,
        vectors_config=models.VectorParams(
            size=app_config.settings.EMBEDDING_MODEL_DIMENSION,
            distance=models.Distance[app_config.settings.EMBEDDING_MODEL_DISTANCE_METRIC.upper()]
        )
    )

@pytest.mark.asyncio
@patch('app.services.qdrant_service.qdrant_client', new_callable=MagicMock)
async def test_get_or_create_collection_fails_on_qdrant_error(mock_global_qdrant_client):
    mock_global_qdrant_client.get_collection.side_effect = UnexpectedResponse(
        status_code=500, content=b"Server error", response=MagicMock()
    )
    collection_name = "failing_collection"

    result = await qdrant_service.get_or_create_collection(collection_name)
    assert result is False
    mock_global_qdrant_client.create_collection.assert_not_called()

@pytest.mark.asyncio
@patch('app.services.qdrant_service.qdrant_client', new_callable=MagicMock)
async def test_list_collections_success(mock_global_qdrant_client):
    # Setup the mock for the collections response structure
    mock_collection1 = MagicMock(spec=models.CollectionDescription)
    mock_collection1.name = "collection1"
    mock_collection2 = MagicMock(spec=models.CollectionDescription)
    mock_collection2.name = "collection2"

    mock_collections_response = MagicMock(spec=models.CollectionsResponse)
    mock_collections_response.collections = [mock_collection1, mock_collection2]

    mock_global_qdrant_client.get_collections.return_value = mock_collections_response # Corrected from list_collections

    result = await qdrant_service.list_collections()

    assert result == ["collection1", "collection2"]
    mock_global_qdrant_client.get_collections.assert_called_once() # Corrected from list_collections

@pytest.mark.asyncio
@patch('app.services.qdrant_service.qdrant_client', new_callable=MagicMock)
async def test_list_collections_failure(mock_global_qdrant_client):
    mock_global_qdrant_client.get_collections.side_effect = Exception("Qdrant unavailable") # Corrected

    result = await qdrant_service.list_collections()

    assert result == []
    mock_global_qdrant_client.get_collections.assert_called_once() # Corrected

# Test for get_qdrant_client function
def test_get_qdrant_client_success():
    # This test assumes the global qdrant_client was successfully initialized (or mocked)
    # If qdrant_service.qdrant_client is None, it should raise ConnectionError
    # To test this properly, you might need to control the initialization of the global client.

    # Scenario 1: Client is initialized (mock it for this test unit)
    with patch('app.services.qdrant_service.qdrant_client', new_callable=MagicMock) as mock_client:
        client_instance = qdrant_service.get_qdrant_client()
        assert client_instance is mock_client

    # Scenario 2: Client is None (failed initialization)
    with patch('app.services.qdrant_service.qdrant_client', None):
        with pytest.raises(ConnectionError, match="Qdrant client is not initialized"):
            qdrant_service.get_qdrant_client()

# Note: To run these tests, ensure pytest and pytest-asyncio are installed.
# poetry run pytest tests/test_qdrant_service.py
# The patching of the global `qdrant_service.qdrant_client` is crucial for these tests
# to work without a running Qdrant instance.
# The mock for `app_config.settings` is also important to control fixed values during tests.
