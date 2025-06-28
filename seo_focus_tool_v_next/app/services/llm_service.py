import httpx
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.models.analysis_models import LlmModel, ProjectAnalyticsResults # LlmSummaryRequest, LlmSummaryResponse
from openai import AsyncOpenAI # For OpenRouter

logger = logging.getLogger(__name__)

openrouter_client = None
if settings.OPENROUTER_API_KEY:
    openrouter_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENROUTER_API_KEY,
    )
else:
    logger.warning("OPENROUTER_API_KEY not set. LLM functionalities will be disabled.")


async def get_openrouter_models() -> List[LlmModel]:
    """
    Fetches the list of available models from OpenRouter.
    """
    if not openrouter_client:
        logger.warning("OpenRouter client not initialized due to missing API key.")
        return []

    # In a real scenario, you might want to cache this list for some time.
    try:
        # The actual OpenRouter API endpoint for models list might differ or need specific client usage.
        # This is based on the plan's `GET https://openrouter.ai/api/v1/models`.
        # The OpenAI client library's `models.list()` is specific to OpenAI's /models endpoint.
        # For OpenRouter, a direct httpx call might be more appropriate if client.models.list() doesn't work.

        # Let's try with a direct httpx call first for fetching models, as client.models.list()
        # might not point to the correct OpenRouter models listing URL or parse its specific format.
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"}
            )
            response.raise_for_status()
            models_data = response.json() # Expects a dictionary with a 'data' key usually

            if "data" in models_data and isinstance(models_data["data"], list):
                return [LlmModel(id=model.get("id"), name=model.get("name", model.get("id"))) for model in models_data["data"]]
            else:
                logger.error(f"Unexpected format from OpenRouter /models endpoint: {models_data}")
                return []

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching OpenRouter models: {e.response.status_code} - {e.response.text}")
        return []
    except Exception as e:
        logger.error(f"Error fetching OpenRouter models: {e}", exc_info=True)
        return []

def format_url_list_for_prompt(urls: List[str], max_urls: int = 5) -> str:
    """Formats a list of URLs for inclusion in the LLM prompt."""
    if not urls:
        return "N/A"
    formatted_list = ""
    for i, url in enumerate(urls[:max_urls], 1):
        formatted_list += f"{i}. {url}\n"
    if len(urls) > max_urls:
        formatted_list += f"... and {len(urls) - max_urls} more\n"
    return formatted_list.strip()

async def generate_summary_openrouter(
    project_name: str,
    model_name: str,
    analysis_results: ProjectAnalyticsResults,
    # Additional context like top focused/divergent URLs might be passed if not in ProjectAnalyticsResults
    # For now, assuming ProjectAnalyticsResults contains enough info or can be augmented.
    top_focused_urls: Optional[List[str]] = None, # These would come from detailed URL analysis
    top_divergent_urls: Optional[List[str]] = None # if not directly in ProjectAnalyticsResults
) -> Optional[str]:
    """
    Generates a summary for the project using a selected OpenRouter LLM.
    """
    if not openrouter_client:
        logger.error("OpenRouter client not available (API key likely missing). Cannot generate summary.")
        return None

    if not analysis_results:
        logger.error(f"No analysis results provided for project {project_name} to generate summary.")
        return None

    # Prepare the prompt (in English, as per requirements)
    # This prompt is an adaptation of the one from the original tool's llm_summarizer.py

    # Ensure scores are available and provide defaults if not
    focus_score_val = analysis_results.site_focus_score if analysis_results.site_focus_score is not None else "N/A"
    radius_score_val = analysis_results.site_radius_score if analysis_results.site_radius_score is not None else "N/A"

    # Cannibalization summary
    cannibalization_summary = "No significant content cannibalization detected."
    if analysis_results.potential_cannibalization:
        num_pairs = len(analysis_results.potential_cannibalization)
        cannibalization_summary = f"{num_pairs} potential content cannibalization pair(s) identified. Top examples:\n"
        for i, pair in enumerate(analysis_results.potential_cannibalization[:3]): # Show top 3
            cannibalization_summary += f"  - '{pair.url1}' and '{pair.url2}' (Similarity: {pair.similarity_score:.2f})\n"

    # For top focused/divergent URLs, these would ideally be derived from a more detailed
    # analysis step that sorts all URLs by distance from centroid.
    # The current ProjectAnalyticsResults doesn't store all URL details.
    # This part needs either:
    # 1. ProjectAnalyticsResults to include top N focused/divergent URLs.
    # 2. This function to fetch all URL details and sort them (less ideal for a summarizer).
    # 3. The caller (API endpoint) to pre-fetch and pass this data.
    # For now, we use placeholders if not directly provided.

    focused_urls_str = format_url_list_for_prompt(top_focused_urls or [])
    divergent_urls_str = format_url_list_for_prompt(top_divergent_urls or [])


    system_prompt = f"""
You are an expert Website Content Strategist and SEO Analyst.
Your task is to analyze the provided data about a website's content structure and topical relevance.
Your audience is the website owner, who may not be an SEO expert.
Your explanation must be clear, concise, and easy to understand, avoiding overly technical jargon.
If technical terms are necessary, explain them simply.
Your goal is to provide a comprehensive summary that includes:
1. An explanation of the key metrics (Focus Score, Radius Score).
2. Insights derived from the data.
3. A practical, actionable plan.

DATA FOR ANALYSIS (Project: {project_name}):

*   Total URLs Analyzed: {analysis_results.total_urls_analyzed}
    *   This is the number of pages on the site included in this analysis.

*   Site Focus Score: {focus_score_val if isinstance(focus_score_val, str) else f'{focus_score_val:.2f}'} / 100
    *   Explanation: Measures how tightly connected the site's content is to a central theme. Higher score = more focused.

*   Site Radius Score: {radius_score_val if isinstance(radius_score_val, str) else f'{radius_score_val:.2f}'} / 100
    *   Explanation: Indicates the breadth of topics covered relative to the main theme. Higher score = wider range of related sub-topics.

*   Most Focused URLs (Examples closest to the site's core theme):
{focused_urls_str}
    *   Insight: These pages likely represent the heart of the website's topic.

*   Most Divergent URLs (Examples furthest from the site's core theme):
{divergent_urls_str}
    *   Insight: These pages are least related to the main topic. Could be necessary (e.g., 'Contact Us') or off-topic.

*   Content Cannibalization Insights:
{cannibalization_summary}
    *   Insight: This highlights pages with very similar content that might compete against each other in search results.

ANALYSIS AND ACTION PLAN REQUIRED:
Please provide the analysis strictly following the structure below. Do not add any introductory sentences before section 1 or concluding remarks after section 3.

1. Executive Summary (Understand Your Site's Topical Health):
    *   Start with a simple, high-level sentence summarizing the site's overall topical focus and breadth.
    *   Explain what the Site Focus Score specifically means for this website in plain English.
    *   Explain what the Site Radius Score specifically means for this website.
    *   Briefly describe how the focus and radius scores work together for this site.

2. Key Insights from Your Content:
    *   Based on the Most Focused URLs, what appears to be the core subject matter of this website?
    *   What do the Most Divergent URLs suggest? Are these expected outliers, or do they dilute focus?
    *   (If Page Type Distribution provided - not available in current ProjectAnalyticsResults model) How does the mix of page types relate to the scores?
    *   What does the content cannibalization data suggest? Are there critical overlaps to address?

3. Action Plan (Steps to Improve or Maintain Focus):
    *   Provide 2-4 specific, actionable recommendations based *directly* on the data analysis.
    *   For each recommendation, state **what** to do and **why** it's suggested.
    *   Phrase recommendations as clear steps.

IMPORTANT:
*   Use straightforward language.
*   Base your analysis *strictly* on the data provided.
*   Deliver ONLY the structured analysis (Sections 1, 2, and 3).
"""

    logger.info(f"Attempting to generate summary for {project_name} using model {model_name} via OpenRouter.")
    try:
        chat_completion = await openrouter_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please provide the analysis for project '{project_name}' based on the data embedded in the system prompt."}
            ],
            # temperature=0.7, # Optional: Adjust creativity
            # max_tokens=1000  # Optional: Limit response length
        )

        if chat_completion.choices and chat_completion.choices[0].message:
            summary_text = chat_completion.choices[0].message.content
            logger.info(f"Successfully generated summary for {project_name} using {model_name}.")
            return summary_text.strip()
        else:
            logger.error(f"OpenRouter returned no summary for {project_name} with model {model_name}. Response: {chat_completion}")
            return None

    except Exception as e:
        logger.error(f"Error generating summary with OpenRouter for {project_name} using {model_name}: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    # Basic test for LLM service
    # Requires OPENROUTER_API_KEY
    logging.basicConfig(level=logging.INFO)

    async def test_llm_runner():
        if not settings.OPENROUTER_API_KEY or settings.OPENROUTER_API_KEY == "your_openrouter_api_key_here":
            logger.warning("OPENROUTER_API_KEY not set. Skipping LLM service test.")
            return

        logger.info("--- Testing LLM Service ---")

        logger.info("Fetching OpenRouter models...")
        models_list = await get_openrouter_models()
        if models_list:
            logger.info(f"Found {len(models_list)} models. First 5:")
            for m in models_list[:5]:
                logger.info(f"  ID: {m.id}, Name: {m.name}")
        else:
            logger.warning("Could not fetch models from OpenRouter.")
            return # Cannot proceed without models

        # Create dummy analysis results for testing summary generation
        dummy_project_name = "llm_test_project"
        dummy_results = ProjectAnalyticsResults(
            project_name=dummy_project_name,
            site_focus_score=75.5,
            site_radius_score=60.2,
            total_urls_analyzed=150,
            potential_cannibalization=[
                CannibalizationPair(url1="http://example.com/pageA", url2="http://example.com/pageB", similarity_score=0.98),
                CannibalizationPair(url1="http://example.com/pageC", url2="http://example.com/pageD", similarity_score=0.96)
            ]
        )
        dummy_focused_urls = ["http://example.com/core-topic-1", "http://example.com/core-topic-2"]
        dummy_divergent_urls = ["http://example.com/contact-us", "http://example.com/privacy-policy"]

        # Use a common, usually free/low-cost model for testing if available
        test_model_id = settings.DEFAULT_LLM_MODEL
        # Check if default model is in the fetched list, or pick one
        available_model_ids = [m.id for m in models_list]
        if test_model_id not in available_model_ids:
            # Try to find a known free model, e.g., Mistral 7B or a GPT 3.5 variant if available
            # This is just for testing robustness.
            candidate_models = [
                "mistralai/mistral-7b-instruct", "openai/gpt-3.5-turbo", "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"
            ]
            for cm in candidate_models:
                if cm in available_model_ids:
                    test_model_id = cm
                    break
            else: # if no candidate found, use first from list if any
                if available_model_ids:
                    test_model_id = available_model_ids[0]
                else:
                    logger.error("No models available from OpenRouter to test summary generation.")
                    return


        logger.info(f"Generating summary for '{dummy_project_name}' using model '{test_model_id}'...")
        summary = await generate_summary_openrouter(
            dummy_project_name,
            test_model_id,
            dummy_results,
            top_focused_urls=dummy_focused_urls,
            top_divergent_urls=dummy_divergent_urls
        )

        if summary:
            logger.info("--- Generated Summary ---")
            logger.info(summary)
            logger.info("--- End of Summary ---")
        else:
            logger.error("Failed to generate summary.")

    asyncio.run(test_llm_runner())
