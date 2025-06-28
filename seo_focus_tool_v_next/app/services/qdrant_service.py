from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Global Qdrant client instance
# It's generally recommended to manage client lifecycle, e.g., within FastAPI app lifespan
# For simplicity here, a global client is used. Consider dependency injection for robust apps.
try:
    qdrant_client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT
        # api_key=settings.QDRANT_API_KEY, # Uncomment if Qdrant is secured with an API key
        # prefer_grpc=True, # For potentially higher performance on large data transfers
    )
    logger.info(f"Successfully connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
except Exception as e:
    logger.error(f"Failed to connect to Qdrant: {e}")
    qdrant_client = None # Set to None if connection fails


def get_qdrant_client() -> QdrantClient:
    """
    Returns the Qdrant client instance.
    Raises an exception if the client is not initialized (e.g., connection failed at startup).
    """
    if qdrant_client is None:
        raise ConnectionError("Qdrant client is not initialized. Check Qdrant server status and connection settings.")
    return qdrant_client

async def get_or_create_collection(collection_name: str) -> bool:
    """
    Checks if a Qdrant collection exists, and creates it if it doesn't.
    The collection name will be sanitized to be Qdrant-compliant.

    Args:
        collection_name (str): The desired name for the collection (e.g., project name).

    Returns:
        bool: True if collection exists or was created successfully, False otherwise.
    """
    client = get_qdrant_client()

    # Qdrant collection names must follow specific rules (e.g., snake_case, no special chars other than _)
    # Here, we'll do a basic sanitization. More robust sanitization might be needed.
    sanitized_collection_name = collection_name.lower().replace(" ", "_").replace("-", "_")
    # Remove any characters not alphanumeric or underscore
    sanitized_collection_name = "".join(c for c in sanitized_collection_name if c.isalnum() or c == '_')
    if not sanitized_collection_name:
        logger.error(f"Collection name '{collection_name}' resulted in an empty sanitized name.")
        return False


    try:
        # Check if collection exists
        # client.get_collection() will raise an exception if not found (or if error)
        try:
            client.get_collection(collection_name=sanitized_collection_name)
            logger.info(f"Collection '{sanitized_collection_name}' already exists.")
            return True
        except UnexpectedResponse as e:
            if e.status_code == 404: # Not found
                 logger.info(f"Collection '{sanitized_collection_name}' not found. Attempting to create.")
            else: # Other error
                raise e # Re-raise other unexpected errors

        # Create collection if it does not exist
        client.create_collection(
            collection_name=sanitized_collection_name,
            vectors_config=models.VectorParams(
                size=settings.EMBEDDING_MODEL_DIMENSION,
                distance=models.Distance[settings.EMBEDDING_MODEL_DISTANCE_METRIC.upper()]
            )
        )
        logger.info(f"Collection '{sanitized_collection_name}' created successfully.")

        # Define payload indexes for fields we might filter/query on
        # Indexing 'url' for efficient checking if a URL has already been processed.
        client.create_payload_index(
            collection_name=sanitized_collection_name,
            field_name="url",
            field_schema=models.PayloadSchemaType.KEYWORD # For exact string matching
        )
        logger.info(f"Payload index created for 'url' field in '{sanitized_collection_name}'.")

        client.create_payload_index(
            collection_name=sanitized_collection_name,
            field_name="processed_at",
            field_schema=models.PayloadSchemaType.DATETIME # For potential time-based queries
        )
        logger.info(f"Payload index created for 'processed_at' field in '{sanitized_collection_name}'.")

        return True

    except UnexpectedResponse as e:
        logger.error(f"Qdrant API error for collection '{sanitized_collection_name}': {e.status_code} - {e.content}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred with collection '{sanitized_collection_name}': {e}")
        return False

async def list_collections() -> list[str]:
    """
    Lists all collections in Qdrant.
    """
    client = get_qdrant_client()
    try:
        collections_response = client.get_collections()
        return [col.name for col in collections_response.collections]
    except Exception as e:
        logger.error(f"Failed to list Qdrant collections: {e}")
        return []

# Example usage (can be part of a test or an admin script)
if __name__ == "__main__":
    import asyncio

    async def main():
        if not qdrant_client:
            print("Qdrant client not initialized. Exiting.")
            return

        # Test get_or_create_collection
        test_collection_name = "my_test_project_123"
        success = await get_or_create_collection(test_collection_name)
        if success:
            print(f"Successfully ensured collection '{test_collection_name}' exists.")
        else:
            print(f"Failed to ensure collection '{test_collection_name}'.")

        # Test listing collections
        collections = await list_collections()
        print(f"\nAvailable collections: {collections}")

        # Clean up (optional)
        # try:
        #     if test_collection_name in collections:
        #         qdrant_client.delete_collection(test_collection_name)
        #         print(f"Cleaned up test collection '{test_collection_name}'.")
        # except Exception as e:
        #     print(f"Error cleaning up test collection: {e}")


    # Configure basic logging for the example
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
