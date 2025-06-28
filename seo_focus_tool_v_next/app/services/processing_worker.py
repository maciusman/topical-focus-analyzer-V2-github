import asyncio
import httpx
import logging
from datetime import datetime, timezone

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
# from app.models.analysis_models import PointPayload # Already defined

# Assuming Jina Embeddings API is compatible with OpenAI client structure
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Configure Jina client (imitating OpenAI client for Jina Embeddings API)
# JINA_EMBEDDINGS_API_URL = "https://api.jina.ai/v1/embeddings" # Defined in plan
# Using openai client means it expects OpenAI compatible API.
# Jina's documentation should be checked for exact compatibility or if a direct httpx call is better.
# For now, proceeding with AsyncOpenAI client structure as per plan's Module 5 for OpenRouter.
# If Jina needs a specific client or direct httpx, this will be adjusted.

jina_openai_client = AsyncOpenAI(
    api_key=settings.JINA_API_KEY,
    base_url="https://api.jina.ai/v1/" # Base URL for Jina, specific endpoint in call
)


async def get_content_from_jina_reader(url: str, client: httpx.AsyncClient) -> str:
    """
    Fetches content from Jina Reader API (https://r.jina.ai/).
    """
    try:
        response = await client.get(
            f"https://r.jina.ai/{url}",
            headers={"Authorization": f"Bearer {settings.JINA_API_KEY}"},
            timeout=60.0 # Increased timeout for reader
        )
        response.raise_for_status()
        # Jina Reader returns content directly, potentially in Markdown if url ends with !md
        # The plan states "Jina Reader API, aby pobrać stronę i przekonwertować ją na czysty format Markdown."
        # Assuming direct output is markdown, or add !md if needed: f"https://r.jina.ai/{url}!md"
        # For now, let's assume it's markdown or good enough text.
        return response.text
    except httpx.HTTPStatusError as e:
        logger.error(f"Jina Reader HTTP error for {url}: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Jina Reader request error for {url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching from Jina Reader for {url}: {e}")
    return "" # Return empty string on error

async def extract_main_content_with_trafilatura(html_or_markdown_content: str) -> str:
    """
    Extracts main content using trafilatura.
    Trafilatura primarily works on HTML, but can sometimes extract from structured text.
    If Jina Reader provides clean Markdown, trafilatura might be less effective or even counterproductive.
    This step needs to be evaluated based on Jina Reader's output.
    For now, assuming it can refine the Markdown or text.
    """
    if not html_or_markdown_content:
        return ""
    try:
        # Import locally if not already globally available or to keep dependencies clear
        import trafilatura
        # favicons=False, ধর্মনিরপেক্ষতা=False are defaults usually.
        # include_comments=False, include_tables=False as per plan.
        # no_fallback=False means it will try to extract something even if main content not clear.
        # target_language can be useful if language is known.
        main_content = trafilatura.extract(
            html_or_markdown_content,
            include_comments=False,
            include_tables=False,
            no_fallback=True # Set to True to get None if no main content found, helps decide if successful
        )
        return main_content if main_content else ""
    except Exception as e:
        logger.error(f"Trafilatura extraction error: {e}")
        return "" # Return empty string or the original content if preferred on error

async def get_embedding_from_jina(text_content: str) -> list[float]:
    """
    Generates embedding for the given text using Jina Embeddings API.
    """
    if not text_content:
        return []
    try:
        # Ensure the model name matches the one specified in the plan and expected by Jina API
        # Plan: "jina-embeddings-v4", task: "text-matching"
        # The OpenAI client might not directly support a 'task' parameter in embeddings.create.
        # This usually is part of the model name or handled by Jina's API endpoint if it's specific.
        # For Jina, custom model names often include task type if needed, or the endpoint itself implies the task.
        # Let's assume "jina-embeddings-v4" is the correct model identifier for the API.
        # If "task" is a separate parameter for Jina's OpenAI-compatible endpoint, this needs verification from Jina docs.
        # For now, proceeding with model name and input.
        response = await jina_openai_client.embeddings.create(
            model="jina-embeddings-v4", # Corrected model name as per plan
            input=[text_content] # API expects a list of strings
            # task="text-matching" # If Jina supports this param via OpenAI client interface
        )
        # Jina API with task "text-matching" might be selected via a different model name string.
        # e.g. model="jina-embeddings-v4-text-matching" - This needs to be verified from Jina's documentation.
        # For now, using the base model name from the plan.

        if response.data and response.data[0].embedding:
            return response.data[0].embedding
        else:
            logger.warning(f"Jina Embeddings API returned no embedding for content snippet: {text_content[:100]}...")
            return []
    except Exception as e:
        logger.error(f"Jina Embeddings API error: {e}")
        return []

def generate_point_id(url: str) -> str:
    """
    Generates a consistent ID for a Qdrant point based on the URL.
    Using a simple hash for now. UUID based on URL could also be an option.
    """
    import hashlib
    return hashlib.sha256(url.encode('utf-8')).hexdigest()


async def process_url_task(
    project_name: str,
    url: str,
    qdrant_cli: QdrantClient,
    http_cli: httpx.AsyncClient,
    status_update_func=None # Callback to update global status
):
    """
    Full processing pipeline for a single URL.
    1. Check if URL already processed in Qdrant for this project.
    2. Fetch content using Jina Reader.
    3. Extract main content using Trafilatura.
    4. Generate embedding using Jina Embeddings API.
    5. Upsert data to Qdrant.
    """
    sanitized_project_name = project_name # Assuming already sanitized by caller
    point_id = generate_point_id(url)

    # 1. Check if URL (via point_id) already processed in Qdrant
    try:
        # Try to retrieve the point by its ID
        # Note: Qdrant's retrieve with a list of IDs is more efficient if checking many.
        # For a single check, this is okay.
        existing_point = qdrant_cli.get_point(collection_name=sanitized_project_name, id=point_id)
        if existing_point:
            logger.info(f"URL {url} (ID: {point_id}) already processed for project {sanitized_project_name}. Skipping.")
            if status_update_func: await status_update_func(project_name, url, "skipped")
            return "skipped"
    except UnexpectedResponse as e:
        if e.status_code == 404: # Not found, which is good, means we need to process it
            logger.debug(f"URL {url} (ID: {point_id}) not found in Qdrant for project {sanitized_project_name}. Proceeding.")
        else: # Other Qdrant error
            logger.error(f"Qdrant error checking point {point_id} for {url} in {sanitized_project_name}: {e}")
            if status_update_func: await status_update_func(project_name, url, "failed", f"Qdrant check error: {e}")
            return "failed"
    except Exception as e: # Other general error during check
        logger.error(f"Error checking point {point_id} for {url} in {sanitized_project_name}: {e}")
        if status_update_func: await status_update_func(project_name, url, "failed", f"General check error: {e}")
        return "failed"

    if status_update_func: await status_update_func(project_name, url, "processing")

    # 2. Fetch content using Jina Reader
    logger.info(f"Fetching content for {url} using Jina Reader...")
    markdown_content = await get_content_from_jina_reader(url, http_cli)
    if not markdown_content:
        logger.warning(f"Failed to get content from Jina Reader for {url}.")
        if status_update_func: await status_update_func(project_name, url, "failed", "Jina Reader failed")
        return "failed_reader"

    # 3. Extract main content using Trafilatura
    logger.info(f"Extracting main content for {url} using Trafilatura...")
    # Jina Reader with !md might already be very clean. Trafilatura might be optional or tuned.
    # If Jina Reader output is Markdown, Trafilatura might not be ideal unless it handles MD well.
    # The plan says "Przekazuje uzyskany Markdown do biblioteki trafilatura".
    cleaned_main_content = await extract_main_content_with_trafilatura(markdown_content)
    if not cleaned_main_content:
        logger.warning(f"Trafilatura extracted no main content for {url}. Using Jina Reader output directly for embedding.")
        # Fallback to using the full markdown from Jina reader if trafilatura returns empty
        # This is a design choice: use potentially noisy Jina output vs. nothing.
        cleaned_main_content = markdown_content


    # 4. Generate embedding using Jina Embeddings API
    logger.info(f"Generating embedding for {url} using Jina Embeddings...")
    # Use the most processed content available for embedding
    content_for_embedding = cleaned_main_content if cleaned_main_content else markdown_content

    vector = await get_embedding_from_jina(content_for_embedding)
    if not vector:
        logger.warning(f"Failed to generate embedding for {url}.")
        if status_update_func: await status_update_func(project_name, url, "failed", "Jina Embeddings failed")
        return "failed_embedding"

    # 5. Upsert data to Qdrant
    try:
        payload = {
            "url": url,
            "content_md": content_for_embedding, # Store the content that was used for embedding
            "processed_at": datetime.now(timezone.utc).isoformat()
        }

        qdrant_cli.upsert(
            collection_name=sanitized_project_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        logger.info(f"Successfully processed and stored {url} (ID: {point_id}) in project {sanitized_project_name}.")
        if status_update_func: await status_update_func(project_name, url, "completed")
        return "completed"
    except Exception as e:
        logger.error(f"Failed to upsert data to Qdrant for {url} (ID: {point_id}): {e}")
        if status_update_func: await status_update_func(project_name, url, "failed", f"Qdrant upsert error: {e}")
        return "failed_qdrant_upsert"

# Example of a status update callback (would be more complex in real app)
# async def simple_status_updater(project, url_processed, status, error=None):
#    print(f"STATUS UPDATE: Project '{project}', URL '{url_processed}', Status: {status}, Error: {error if error else 'None'}")

if __name__ == "__main__":
    # Basic test for process_url_task
    # This requires a running Qdrant instance and valid JINA_API_KEY
    # Also, `trafilatura` needs to be installed.
    logging.basicConfig(level=logging.INFO)

    async def test_runner():
        # Ensure Qdrant client is available (it's global in qdrant_service)
        try:
            qdrant_service.get_qdrant_client().list_collections() # Test connection
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}. Aborting test.")
            return

        test_project = "test_processing_worker_project"
        test_url = "https://www.python.org/about/gettingstarted/" # A relatively simple page

        # Create collection for testing
        await qdrant_service.get_or_create_collection(test_project)

        async with httpx.AsyncClient() as http_client_for_test:
            result = await process_url_task(
                test_project,
                test_url,
                qdrant_service.get_qdrant_client(),
                http_client_for_test,
                # status_update_func=simple_status_updater
            )
            logger.info(f"Test processing for {test_url} resulted in: {result}")

            # Verify in Qdrant
            if result == "completed":
                point_id_test = generate_point_id(test_url)
                try:
                    retrieved = qdrant_service.get_qdrant_client().get_point(test_project, point_id_test)
                    logger.info(f"Retrieved point from Qdrant: {retrieved.id}, payload keys: {list(retrieved.payload.keys()) if retrieved.payload else 'None'}")
                except Exception as e:
                    logger.error(f"Failed to retrieve test point: {e}")

        # Clean up (optional)
        # try:
        #     qdrant_service.get_qdrant_client().delete_collection(test_project)
        #     logger.info(f"Cleaned up test collection '{test_project}'.")
        # except Exception as e:
        #     logger.error(f"Error cleaning up test collection '{test_project}': {e}")


    if settings.JINA_API_KEY and settings.JINA_API_KEY != "your_jina_api_key_here":
        asyncio.run(test_runner())
    else:
        logger.warning("JINA_API_KEY not set. Skipping process_url_task test.")
