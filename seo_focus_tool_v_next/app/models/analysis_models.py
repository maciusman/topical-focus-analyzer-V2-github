from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime

class UrlStatus(BaseModel):
    url: HttpUrl
    status: str # e.g., "pending", "processing", "completed", "failed"
    error_message: Optional[str] = None

class ProjectStatus(BaseModel):
    project_name: str
    total_urls: int
    processed_urls: int
    pending_urls: int
    failed_urls: int
    urls_status: List[UrlStatus] = []

class AnalysisRequest(BaseModel):
    project_name: str
    # urls: List[HttpUrl] # URLs will be read from CSV by the server

class PointPayload(BaseModel):
    url: str # Storing as str as HttpUrl can be slow for large lists and Qdrant stores it as string
    content_md: str
    processed_at: datetime
    # Potentially other metadata in the future
    # title: Optional[str] = None
    # lang: Optional[str] = None


# --- Models for Results (Module 4) ---
# These are preliminary and will be refined in Module 4

class UrlAnalysisDetail(BaseModel):
    url: str
    content_preview: Optional[str] = None # First few characters of content_md
    vector_id: Optional[str] = None # Qdrant point ID
    x_coord: Optional[float] = None # For UMAP/t-SNE
    y_coord: Optional[float] = None # For UMAP/t-SNE
    z_coord: Optional[float] = None # For 3D UMAP/t-SNE
    distance_from_centroid: Optional[float] = None
    # Add other relevant fields from existing app: page_type, page_depth, source_sitemap (though sitemap not used as input now)

class CannibalizationPair(BaseModel):
    url1: str
    url2: str
    similarity_score: float # Cosine similarity
    # content_preview1: Optional[str] = None
    # content_preview2: Optional[str] = None

class ProjectAnalyticsResults(BaseModel):
    project_name: str
    site_focus_score: Optional[float] = None
    site_radius_score: Optional[float] = None
    total_urls_analyzed: int
    centroid_vector: Optional[List[float]] = None # Might be large, consider if needed by client
    # url_details: List[UrlAnalysisDetail] = [] # This could be very large, might need pagination or separate endpoint
    potential_cannibalization: List[CannibalizationPair] = []
    # ai_summary: Optional[str] = None # This will come from LLM service

class LlmModel(BaseModel):
    id: str
    name: Optional[str] = None # OpenRouter provides more details, we can include them

class LlmSummaryRequest(BaseModel):
    model_name: str
    # analysis_results: ProjectAnalyticsResults # Pass relevant parts, not necessarily the whole thing

class LlmSummaryResponse(BaseModel):
    project_name: str
    model_used: str
    summary_text: str
    generated_at: datetime
