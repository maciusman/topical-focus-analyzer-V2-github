import streamlit as st
import pandas as pd
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration for the FastAPI backend
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")

# Helper function to make requests to FastAPI
async def api_request(method: str, endpoint: str, **kwargs):
    async with httpx.AsyncClient(base_url=FASTAPI_BASE_URL) as client:
        try:
            response = await client.request(method, endpoint, **kwargs)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()
        except httpx.HTTPStatusError as e:
            st.error(f"API Error: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            st.error(f"Request Error: Failed to connect to API at {FASTAPI_BASE_URL}{endpoint}. Ensure the backend is running. Error: {e}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None

# --- UI Layout ---
st.set_page_config(page_title="SEO Focus Tool v-Next", layout="wide")
st.title("🔍 SEO Focus & Radius Analyzer v-Next")

# --- Sidebar for Project Management and LLM Selection ---
with st.sidebar:
    st.header("Project Setup")
    project_name = st.text_input("Project Name", placeholder="e.g., my-website-analysis")

    st.header("LLM Selection (via OpenRouter)")
    # Placeholder for LLM models - will be fetched from API
    # For now, using a static list or allowing text input
    # In future, this will call an endpoint like `/llm/models`
    # llm_models_data = await api_request("GET", "/llm/models")
    # available_llm_models = [model.get('id') for model in llm_models_data] if llm_models_data else ["(Not loaded)"]

    # Temporary solution for LLM model selection:
    st.info("LLM model list will be dynamically loaded from OpenRouter via API in a future step.")
    selected_llm_model = st.text_input("Enter OpenRouter Model ID", value="openai/gpt-3.5-turbo", help="e.g., openai/gpt-3.5-turbo, google/gemini-pro")


# --- Main Area for Analysis and Results ---
tab_titles = ["New Analysis", "Load Project", "Results Overview", "URL Details", "Visual Map", "Cannibalization"]
tabs = st.tabs(tab_titles)

with tabs[0]: # New Analysis
    st.subheader("Start a New Analysis")
    uploaded_file = st.file_uploader("Upload CSV with URLs (one URL per line, first column)", type=["csv"])

    start_button = st.button("Start Analysis", key="start_analysis_button", disabled=not (project_name and uploaded_file and selected_llm_model))

    if start_button:
        if not project_name:
            st.warning("Please enter a Project Name.")
        elif not uploaded_file:
            st.warning("Please upload a CSV file with URLs.")
        elif not selected_llm_model:
            st.warning("Please select or enter an LLM Model ID.")
        else:
            st.info(f"Starting analysis for project: {project_name} with model {selected_llm_model}")
            # TODO:
            # 1. Read CSV
            # 2. Send data to FastAPI endpoint `/analysis/start`
            #    - project_name
            #    - file
            #    - selected_llm_model (if needed at start or later for summary)
            # 3. Start listening to SSE for progress updates from `/analysis/status/{project_name}`

            # Example of sending file to backend (conceptual)
            # files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            # data = {"project_name": project_name}
            # response = await api_request("POST", "/analysis/start", files=files, data=data)
            # if response:
            #    st.success("Analysis started successfully! Check progress below.")
            #    # Setup SSE listener here

            st.markdown("---")
            st.subheader("Analysis Progress")
            progress_placeholder = st.empty()
            progress_placeholder.info("Waiting for analysis to start...")
            # TODO: Implement SSE client to update progress_placeholder

            # Example:
            # async def stream_progress(project_name_for_sse):
            #     async with httpx.AsyncClient() as client:
            #         async with client.stream("GET", f"{FASTAPI_BASE_URL}/analysis/status/{project_name_for_sse}") as response:
            #             if response.status_code == 200:
            #                 async for line in response.aiter_lines():
            #                     if line.startswith("data:"):
            #                         message = line[len("data:"):]
            #                         progress_placeholder.info(message)
            #             else:
            #                 progress_placeholder.error(f"Failed to connect to progress stream: {response.status_code}")
            # asyncio.run(stream_progress(project_name))


with tabs[1]: # Load Project
    st.subheader("Load Existing Project")
    # TODO:
    # 1. Fetch list of existing projects from FastAPI endpoint `/projects`
    # 2. Display in a selectbox
    # 3. On selection, load results for that project by calling `/analysis/results/{project_name}`
    #    and potentially `/llm/summarize/{project_name}` if summary not part of main results.

    # Placeholder
    # projects_data = await api_request("GET", "/projects")
    # available_projects = [proj.get('name') for proj in projects_data] if projects_data else ["(No projects found)"]
    # selected_project_to_load = st.selectbox("Select Project to Load", available_projects)
    # load_project_button = st.button("Load Project", key="load_project_button")

    # if load_project_button and selected_project_to_load and selected_project_to_load != "(No projects found)":
    #    st.info(f"Loading project: {selected_project_to_load}")
        # results = await api_request("GET", f"/analysis/results/{selected_project_to_load}")
        # if results:
        #    st.session_state['current_project_results'] = results # Store in session state
        #    st.success(f"Project '{selected_project_to_load}' loaded. View results in other tabs.")
        #    # Potentially switch to Results Overview tab
    st.info("Functionality to load existing projects will be implemented here.")


with tabs[2]: # Results Overview
    st.subheader("Results Overview")
    # TODO: Display Focus Score, Radius Score, AI Summary, Charts (Page Type, Sitemap Dist)
    # Data will come from st.session_state['current_project_results']
    st.info("Key metrics, AI summary, and overview charts will be displayed here once a project is analyzed or loaded.")

with tabs[3]: # URL Details
    st.subheader("URL Analysis Details")
    # TODO: Display table of URLs with their metrics (distance, coords, etc.)
    # Filters for search, page type, etc.
    st.info("A detailed table of all analyzed URLs with their individual metrics and content previews will be available here.")

with tabs[4]: # Visual Map
    st.subheader("Topical Map Visualization (UMAP/t-SNE)")
    # TODO: Display Plotly interactive scatter plot
    st.info("An interactive 2D or 3D map (UMAP or t-SNE) visualizing the topical relationships between URLs will be shown here.")

with tabs[5]: # Cannibalization
    st.subheader("Content Cannibalization / Clusters")
    # TODO: Display pairs of URLs with high similarity, content previews
    st.info("Potential content cannibalization issues (highly similar URLs) and thematic clusters will be identified and displayed here.")


# --- Footer ---
st.markdown("---")
st.markdown("SEO Focus Tool v-Next | Powered by FastAPI, Streamlit, Jina AI, Qdrant, and OpenRouter")

# To run this Streamlit app:
# Ensure FastAPI backend (app.main:app) is running.
# Then, in a new terminal in the `seo_focus_tool_v_next` directory:
# poetry run streamlit run streamlit_app.py
#
# For async operations within Streamlit event handlers (like button clicks),
# you might need to use asyncio.run() or ensure Streamlit's event loop
# is compatible if running Streamlit > 1.17 which has better native asyncio support.
# For simplicity in this skeleton, direct `await` calls are shown,
# but may need adjustment based on Streamlit version and context.
# A common pattern is:
# if st.button("..."):
#     result = asyncio.run(api_request(...))
#     if result: ...
# This will be refined as specific API calls are implemented.
