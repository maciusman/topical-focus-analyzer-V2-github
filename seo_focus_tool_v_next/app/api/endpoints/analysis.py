from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
import csv
import io
import httpx
import json # Added for SSE
from typing import List, Dict, Any, Optional
import logging
from contextlib import asynccontextmanager

from app.services import qdrant_service
from app.services import processing_worker
from app.models.analysis_models import ProjectStatus, UrlStatus # Using the Pydantic models
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# --- In-memory storage for POC ---
# For a production app, use Redis or a proper task queue (e.g., Celery, RabbitMQ)
# and a persistent way to track job status.
tasks_queue: asyncio.Queue = asyncio.Queue()
project_status_db: Dict[str, ProjectStatus] = {}
active_workers: List[asyncio.Task] = []
MAX_WORKERS = settings.MAX_WORKERS if hasattr(settings, 'MAX_WORKERS') else 3 # Default to 3 if not in settings
PROCESSING_SEMAPHORE = asyncio.Semaphore(MAX_WORKERS) # Limit concurrent external API calls

# --- Helper for status updates ---
async def update_url_status(project_name: str, url: str, status: str, error_message: Optional[str] = None):
    if project_name not in project_status_db:
        logger.warning(f"Project {project_name} not found in status_db for status update of {url}.")
        return

    # Find and update specific URL status
    updated_url_in_list = False
    for url_stat_obj in project_status_db[project_name].urls_status:
        if url_stat_obj.url == url:
            url_stat_obj.status = status
            url_stat_obj.error_message = error_message
            updated_url_in_list = True
            break
    if not updated_url_in_list: # Add if not present (should be added on init)
        project_status_db[project_name].urls_status.append(UrlStatus(url=url, status=status, error_message=error_message))


    # Update aggregate counts
    if status == "completed" or status == "skipped":
        project_status_db[project_name].processed_urls += 1
        project_status_db[project_name].pending_urls -=1
    elif status == "failed":
        project_status_db[project_name].failed_urls += 1
        project_status_db[project_name].pending_urls -=1
    # "processing" status does not change aggregate counts here, it's an intermediate state.

    logger.info(f"Status updated for {url} in {project_name}: {status}. Pending: {project_status_db[project_name].pending_urls}")


# --- Worker Implementation ---
async def worker_main(name: str, queue: asyncio.Queue, q_client: qdrant_service.QdrantClient):
    logger.info(f"Worker {name} started.")
    async with httpx.AsyncClient(timeout=None) as http_client: # Client for Jina, OpenRouter
        while True:
            try:
                project_name, url = await queue.get()
                logger.info(f"Worker {name} processing: {url} for project {project_name}")

                async with PROCESSING_SEMAPHORE: # Acquire semaphore before processing
                    await processing_worker.process_url_task(
                        project_name,
                        url,
                        q_client,
                        http_client,
                        status_update_func=update_url_status # Pass the callback
                    )

                queue.task_done()
            except asyncio.CancelledError:
                logger.info(f"Worker {name} was cancelled.")
                break
            except Exception as e:
                logger.error(f"Worker {name} encountered an error: {e}", exc_info=True)
                # Attempt to update status to failed if possible
                if 'project_name' in locals() and 'url' in locals():
                     await update_url_status(project_name, url, "failed", str(e))
                queue.task_done() # Ensure task_done is called even on error


# --- FastAPI Lifespan Management for Workers ---
@asynccontextmanager
async def lifespan(app: APIRouter): # Changed from app: FastAPI to app: APIRouter for router specific lifespan
    # Startup
    logger.info(f"FastAPI application startup: Starting {MAX_WORKERS} processing workers...")
    q_client = qdrant_service.get_qdrant_client() # Get client once
    if q_client:
        for i in range(MAX_WORKERS):
            task = asyncio.create_task(worker_main(f"worker-{i+1}", tasks_queue, q_client))
            active_workers.append(task)
        logger.info(f"{len(active_workers)} Processing workers started.")
    else:
        logger.error("Qdrant client not available. Workers not started.")

    yield # Application runs here

    # Shutdown
    logger.info("FastAPI application shutdown: Shutting down workers...")
    if tasks_queue.qsize() > 0:
        logger.info(f"Waiting for {tasks_queue.qsize()} items in queue to be processed...")
        await tasks_queue.join() # Wait for queue to be empty
        logger.info("All items processed.")

    for task in active_workers:
        if not task.done():
            task.cancel()

    # Allow time for tasks to acknowledge cancellation
    # Gather results, including potential CancelledError exceptions
    results = await asyncio.gather(*active_workers, return_exceptions=True)
    for i, result in enumerate(results):
        if isinstance(result, asyncio.CancelledError):
            logger.info(f"Worker worker-{i+1} successfully cancelled.")
        elif isinstance(result, Exception):
            logger.error(f"Worker worker-{i+1} raised an exception during shutdown: {result}")

    logger.info("Processing workers shut down.")

# Apply lifespan to the router
router.router.lifespan_context = lifespan


@router.post("/start", summary="Start New Analysis")
async def start_analysis(
    project_name: str = Form(..., description="Name of the project for this analysis. Will be used as Qdrant collection name."),
    file: UploadFile = File(..., description="CSV file containing a list of URLs to analyze. Each URL should be in the first column of a new row.")
):
    """
    Starts a new analysis for the given project name and CSV file of URLs.
    - Creates a new Qdrant collection if it doesn't exist.
    - Reads URLs from the CSV.
    - Checks Qdrant for already processed URLs for this project.
    - Adds new URLs to an asynchronous processing queue.
    """
    logger.info(f"Received request to start analysis for project: {project_name}")

    # Sanitize project_name for Qdrant
    sanitized_project_name = project_name.lower().replace(" ", "_").replace("-", "_")
    sanitized_project_name = "".join(c for c in sanitized_project_name if c.isalnum() or c == '_')

    if not sanitized_project_name:
        raise HTTPException(status_code=400, detail="Invalid project name. After sanitization, the name is empty.")

    # 1. Get or create Qdrant collection
    collection_created_or_exists = await qdrant_service.get_or_create_collection(sanitized_project_name)
    if not collection_created_or_exists:
        raise HTTPException(status_code=500, detail=f"Failed to create or access Qdrant collection for project '{sanitized_project_name}'.")

    # 2. Read URLs from CSV
    urls_to_process = []
    try:
        content = await file.read()
        # Assuming UTF-8 encoding, adjust if other encodings are expected
        csv_content = content.decode('utf-8')
        reader = csv.reader(io.StringIO(csv_content))
        for row in reader:
            if row: # Ensure row is not empty
                url = row[0].strip()
                if url: # Ensure URL is not empty
                    # Basic URL validation could be added here (e.g., using Pydantic's HttpUrl)
                    urls_to_process.append(url)
        logger.info(f"Read {len(urls_to_process)} URLs from CSV for project '{sanitized_project_name}'.")
    except Exception as e:
        logger.error(f"Error reading or parsing CSV for project '{sanitized_project_name}': {e}")
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {e}")
    finally:
        await file.close()

    if not urls_to_process:
        return JSONResponse(status_code=200, content={"message": "No URLs found in the provided CSV file.", "project_name": sanitized_project_name, "urls_added_to_queue": 0})

    # 3. Check Qdrant for already processed URLs & add new ones to queue (Simplified - actual checking will be in worker)
    # For this stage, we'll just add all unique URLs from CSV to the queue.
    # The worker will be responsible for the "get or process" logic.

    unique_urls = set(urls_to_process) # Process only unique URLs from the CSV
    initial_url_statuses = [UrlStatus(url=u, status="pending") for u in unique_urls]

    # Initialize project status
    project_status_db[sanitized_project_name] = ProjectStatus(
        project_name=sanitized_project_name,
        total_urls=len(unique_urls),
        processed_urls=0,
        pending_urls=len(unique_urls),
        failed_urls=0,
        urls_status=initial_url_statuses
    )

    added_to_queue_count = 0
    for url_to_process_item in unique_urls: # Iterate over the set of unique URLs
        # We simply add the task to the queue.
        await tasks_queue.put((sanitized_project_name, url_to_process_item))
        added_to_queue_count += 1

    logger.info(f"Added {added_to_queue_count} URLs to the processing queue for project '{sanitized_project_name}'. Initial status created.")

    return JSONResponse(
        status_code=202, # Accepted
        content={
            "message": "Analysis request accepted. URLs are being added to the processing queue.",
            "project_name": sanitized_project_name,
            "total_unique_urls_in_csv": len(unique_urls),
            "urls_added_to_queue": added_to_queue_count
        }
    )


# Placeholder for SSE status endpoint (to be implemented later)
# @router.get("/status/{project_name}", summary="Get Analysis Status (SSE)")
# async def get_analysis_status(project_name: str):
#     """
#     Streams the analysis status for a given project using Server-Sent Events (SSE).
#     """
#     sanitized_project_name = project_name.lower().replace(" ", "_").replace("-", "_")
#     sanitized_project_name = "".join(c for c in sanitized_project_name if c.isalnum() or c == '_')

#     async def event_generator():
#         # This is a simplified generator. A real implementation would check a shared status more robustly.
#         last_processed_count = -1
#         while True:
#             if sanitized_project_name in project_status_db:
#                 status = project_status_db[sanitized_project_name]
#                 if status.processed_urls != last_processed_count or status.pending_urls == 0:
#                     yield f"data: Processed {status.processed_urls}/{status.total_urls}, Pending: {status.pending_urls}, Failed: {status.failed_urls}\n\n"
#                     last_processed_count = status.processed_urls
#                     if status.pending_urls == 0 and tasks_queue.empty(): # Basic check if queue is also empty for this project
#                         yield f"data: Processing complete for project {sanitized_project_name}.\n\n"
#                         break
#             else:
#                 yield f"data: Project '{sanitized_project_name}' not found or not yet initialized.\n\n"
#                 break # Stop if project status doesn't exist

#             await asyncio.sleep(2) # Send update every 2 seconds

#     return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/projects", summary="List All Projects (Collections)")
async def list_projects():
    """
    Lists all existing projects (Qdrant collections).
    """
    collections = await qdrant_service.list_collections()
    return {"projects": collections}

@router.get("/status/{project_name}", summary="Get Analysis Status", response_model=ProjectStatus)
async def get_analysis_status(project_name: str):
    """
    Retrieves the current processing status for a given project.
    This is a simple GET endpoint; SSE will be implemented for real-time updates.
    """
    sanitized_project_name = project_name.lower().replace(" ", "_").replace("-", "_")
    sanitized_project_name = "".join(c for c in sanitized_project_name if c.isalnum() or c == '_')

    if sanitized_project_name in project_status_db:
        # Ensure pending_urls is not negative if total_urls was updated after some processing
        status = project_status_db[sanitized_project_name]
        actual_pending = status.total_urls - status.processed_urls - status.failed_urls
        status.pending_urls = max(0, actual_pending)
        return status
    else:
        # If project not in DB, it might mean it's not started or an invalid name
        # Check if a collection exists in Qdrant to give a more informed message
        try:
            q_client = qdrant_service.get_qdrant_client()
            q_client.get_collection(collection_name=sanitized_project_name)
            # Collection exists but not in our in-memory status DB (e.g., after server restart)
            # For now, we can't reconstruct detailed status without persistent job tracking.
            # A real system would initialize status from a DB or task queue metadata.
            raise HTTPException(status_code=404, detail=f"Project '{sanitized_project_name}' exists but its live status is not tracked. Analysis might be complete or server restarted.")
        except HTTPException: # Re-raise the HTTPException above
            raise
        except Exception: # Qdrant get_collection raised something else (e.g. not found)
            raise HTTPException(status_code=404, detail=f"Project '{sanitized_project_name}' not found or not initiated.")


@router.get("/status/sse/{project_name}", summary="Get Analysis Status (SSE)")
async def get_analysis_status_sse(project_name: str):
    """
    Streams the analysis status for a given project using Server-Sent Events (SSE).
    """
    sanitized_project_name = project_name.lower().replace(" ", "_").replace("-", "_")
    sanitized_project_name = "".join(c for c in sanitized_project_name if c.isalnum() or c == '_')

    async def event_generator():
        logger.info(f"SSE connection opened for project: {sanitized_project_name}")
        last_sent_status_summary = "" # To send updates only on change

        try:
            while True:
                if sanitized_project_name not in project_status_db:
                    status_data = {"error": f"Project '{sanitized_project_name}' not found or not initialized."}
                    yield f"data: {json.dumps(status_data)}\n\n"
                    break # Stop streaming if project is gone

                status = project_status_db[sanitized_project_name]
                # Ensure pending is correctly calculated if other counts change
                actual_pending = status.total_urls - status.processed_urls - status.failed_urls
                status.pending_urls = max(0, actual_pending)

                # Create a summary of the current status for comparison
                current_status_summary = f"Processed: {status.processed_urls}, Pending: {status.pending_urls}, Failed: {status.failed_urls}, Total: {status.total_urls}"

                if current_status_summary != last_sent_status_summary:
                    # Send the full ProjectStatus model as JSON
                    status_payload = status.model_dump_json() # Use Pydantic's json export
                    yield f"data: {status_payload}\n\n"
                    last_sent_status_summary = current_status_summary

                if status.pending_urls == 0 and status.processed_urls + status.failed_urls == status.total_urls:
                    # This condition means all initial URLs have reached a terminal state (completed, skipped, or failed)
                    # Check if the task queue is also empty for this project (more complex, not directly checkable here)
                    # For now, if pending is 0, we assume processing for *this batch* is done.
                    final_message = {"message": f"Processing complete for project {sanitized_project_name}.", "status": status.model_dump()}
                    yield f"data: {json.dumps(final_message)}\n\n"
                    logger.info(f"SSE stream: Processing deemed complete for {sanitized_project_name}. Sent final message.")
                    break

                await asyncio.sleep(settings.SSE_UPDATE_INTERVAL if hasattr(settings, 'SSE_UPDATE_INTERVAL') else 2) # Send update interval
        except asyncio.CancelledError:
            logger.info(f"SSE connection cancelled by client for project: {sanitized_project_name}")
        except Exception as e:
            logger.error(f"Error in SSE generator for {sanitized_project_name}: {e}", exc_info=True)
            error_data = {"error": f"An error occurred in the SSE stream: {str(e)}"}
            try:
                yield f"data: {json.dumps(error_data)}\n\n"
            except Exception: # Handle cases where yield might fail (e.g. client disconnected)
                pass
        finally:
            logger.info(f"SSE connection closed for project: {sanitized_project_name}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Include this router in app/main.py
# from .api.endpoints import analysis
# app.include_router(analysis.router, prefix="/analysis", tags=["Analysis Processing"])
