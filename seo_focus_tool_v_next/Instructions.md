# SEO Focus Tool v-Next

## Project Overview

This project is a rebuilt and enhanced version of a Topical Focus Analyzer tool. It aims to provide SEO insights by analyzing the **Site Focus Score** and **Site Radius Score** for a given list of URLs. The tool leverages modern technologies including Jina AI for content processing and embeddings, Qdrant for vector storage, FastAPI for the backend API, and Streamlit for the user interface.

The core idea is to determine how thematically focused a website (or a list of its pages) is, and what the breadth of its topical coverage is.

## Key Features (Planned & Implemented)

*   **CSV Input:** Users provide a list of URLs via a CSV file.
*   **Content Processing:**
    *   Jina Reader API: Fetches URL content and converts to Markdown.
    *   `trafilatura`: Extracts the main article content from the Markdown.
*   **Embedding Generation:**
    *   Jina Embeddings API (`jina-embeddings-v4`): Generates high-quality vector embeddings for the cleaned content.
*   **Vector Storage:**
    *   Qdrant: Stores and indexes the generated embeddings for efficient similarity searches and analysis.
*   **Backend API:**
    *   FastAPI: Provides a robust, asynchronous API for managing analysis tasks, retrieving results, and interacting with LLMs.
    *   Background Workers: Process URLs asynchronously.
    *   SSE: Real-time progress updates for the analysis.
*   **Core Analytics:**
    *   **Site Focus Score:** Measures how tightly content clusters around a central theme.
    *   **Site Radius Score:** Measures the breadth of topics covered.
    *   **Cannibalization Detection:** Identifies URLs with highly similar content.
*   **AI-Powered Summaries:**
    *   OpenRouter Integration: Allows dynamic selection of various LLMs to generate summaries and insights based on the analysis data.
*   **Visualization:**
    *   UMAP/t-SNE: Dimensionality reduction for visualizing content clusters (via Plotly in Streamlit).
*   **User Interface:**
    *   Streamlit: Provides an interactive dashboard for uploading URLs, managing projects, viewing results, and interacting with visualizations.

## Tech Stack

*   **Python 3.11+**
*   **Poetry:** For dependency management.
*   **FastAPI:** For the backend API framework.
*   **Uvicorn:** ASGI server for FastAPI.
*   **Streamlit:** For the user interface.
*   **Qdrant:** Vector database.
*   **Jina AI:**
    *   Reader API (via `https://r.jina.ai/`)
    *   Embeddings API (`jina-embeddings-v4` via `https://api.jina.ai/v1/embeddings`)
*   **`trafilatura`:** For main content extraction.
*   **OpenRouter:** For access to various LLMs.
*   **`httpx`:** Asynchronous HTTP client.
*   **`scikit-learn` & `umap-learn`:** For analytics and dimensionality reduction.
*   **Plotly:** For interactive visualizations.
*   **Docker:** For running Qdrant.
*   **`python-dotenv`:** For managing environment variables.
*   **`pytest` & `pytest-asyncio`:** For testing.

## Setup and Running

### Prerequisites

1.  **Git:** Ensure Git is installed on your system. You can download it from [git-scm.com](https://git-scm.com/).
2.  **Python:** Install Python version 3.11 or newer. You can download it from [python.org](https://www.python.org/). During installation, make sure to check the box that says "Add Python to PATH".
3.  **Poetry:** This tool manages project dependencies. After installing Python, open your terminal (Command Prompt, PowerShell, or Terminal on Mac/Linux) and install Poetry by running:
    ```bash
    pip install poetry
    ```
    If `pip` is not recognized, you might need to use `python -m pip install poetry` or ensure Python's Scripts directory is in your PATH.
4.  **Docker Desktop:** Qdrant (the vector database) runs in a Docker container. Download and install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/). Make sure Docker Desktop is running before you proceed to run Qdrant.

### 1. Download the Project Files

Since the files are on GitHub, you have two main ways to get them:

*   **Option A: Clone the Repository (Recommended if you have Git)**
    1.  Open your terminal.
    2.  Navigate to the directory where you want to store the project (e.g., `cd Documents/Projects`).
    3.  Clone the repository using its HTTPS URL (you'll find this URL on the GitHub page of the repository):
        ```bash
        git clone https://github.com/TWOJA_NAZWA_UŻYTKOWNIKA/NAZWA_REPOZYTORIUM.git
        ```
        (Replace `TWOJA_NAZWA_UŻYTKOWNIKA/NAZWA_REPOZYTORIUM` with the actual path to the repository).
    4.  Navigate into the cloned project directory:
        ```bash
        cd NAZWA_REPOZYTORIUM/seo_focus_tool_v_next
        ```
        (Or directly `cd seo_focus_tool_v_next` if you cloned that specific subfolder, though typically you clone the whole repo).

*   **Option B: Download as ZIP**
    1.  On the GitHub page of the repository, look for a green button labeled "Code".
    2.  Click it, and then click "Download ZIP".
    3.  Once downloaded, extract the ZIP file to a location of your choice.
    4.  Open your terminal and navigate into the extracted folder, specifically into the `seo_focus_tool_v_next` directory.

### 2. Configure Environment Variables (API Keys)

This application requires API keys for Jina AI and OpenRouter.

1.  **Locate the `.env` file:** Inside the `seo_focus_tool_v_next` directory, you should find a file named `.env`. If it's not there, or if there's a file like `.env.example`, create/rename it to `.env`.
2.  **Edit `.env`:** Open this file with a simple text editor (like Notepad, VS Code, Sublime Text, etc.).
3.  **Add your API keys:** The file should look something like this. Replace the placeholder text with your actual API keys:
    ```env
    # .env
    JINA_API_KEY="your_jina_api_key_here"
    OPENROUTER_API_KEY="your_openrouter_api_key_here"
    
    # Optional: Qdrant connection details if not using localhost default
    # QDRANT_HOST="localhost"
    # QDRANT_PORT="6333"
    
    # Optional: Default LLM model for OpenRouter
    # DEFAULT_LLM_MODEL="openai/gpt-3.5-turbo"
    
    # Optional: Max concurrent processing workers for URL fetching/embedding
    # MAX_WORKERS=3
    
    # Optional: SSE Update Interval in seconds for progress streaming
    # SSE_UPDATE_INTERVAL=2
    ```
    *   Get your Jina API key from the [Jina AI Dashboard](https://cloud.jina.ai/).
    *   Get your OpenRouter API key from the [OpenRouter Dashboard](https://openrouter.ai/keys).
4.  **Save the `.env` file.** This file is listed in `.gitignore`, so your keys will not be accidentally committed to GitHub if you decide to use Git for your own versioning.

### 3. Install Project Dependencies

This step uses Poetry to install all the Python libraries the project needs.

1.  Make sure you are in the `seo_focus_tool_v_next` directory in your terminal.
2.  Run the following command:
    ```bash
    poetry install
    ```
    Poetry will read the `pyproject.toml` file, resolve all dependencies, and install them into a dedicated virtual environment. This might take a few minutes the first time.

### 4. Run Qdrant Vector Database

Qdrant is used to store the data (vector embeddings) for your analysis.

1.  Ensure Docker Desktop is running on your computer.
2.  Open a new terminal window (leave the current one for other commands).
3.  Run the following command to start Qdrant:
    ```bash
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    ```
    This command downloads the Qdrant image (if you don't have it) and starts a container. Port `6333` is for the HTTP API and `6334` for gRPC. The application will connect to Qdrant on `localhost:6333` by default.
    Keep this terminal window open; Qdrant will be running in it.

### 5. Run the FastAPI Backend Application

The backend handles the data processing and analysis logic.

1.  Open another new terminal window.
2.  Navigate to the `seo_focus_tool_v_next` directory.
3.  Activate the Poetry virtual environment. If you are not already in it (e.g., if your terminal prompt doesn't show the environment name), type:
    ```bash
    poetry shell
    ```
    This command activates the virtual environment that Poetry created for the project.
4.  Now, start the FastAPI server:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `app.main:app` tells Uvicorn where to find the FastAPI application instance (in the `main.py` file inside the `app` directory).
    *   `--reload` makes the server restart automatically if you make code changes (useful for development).
    *   `--host 0.0.0.0` makes the server accessible from your local network (optional, `localhost` is default).
    *   `--port 8000` specifies the port it runs on.
    You should see output indicating the server is running, e.g., `Uvicorn running on http://0.0.0.0:8000`. Keep this terminal open.
    You can check if the API is running by opening `http://localhost:8000/health` in your web browser. You should see `{"status":"healthy","message":"API is running!"}`. The API documentation will be at `http://localhost:8000/docs`.

### 6. Run the Streamlit User Interface

The Streamlit application provides the web interface to interact with the tool.

1.  Open a *fourth* new terminal window (yes, you might have a few open now! Qdrant, FastAPI, and this one).
2.  Navigate to the `seo_focus_tool_v_next` directory.
3.  Activate the Poetry virtual environment if you haven't already in this terminal:
    ```bash
    poetry shell
    ```
4.  Start the Streamlit application:
    ```bash
    streamlit run streamlit_app.py
    ```
    Streamlit will usually open the application automatically in your default web browser (typically at `http://localhost:8501`). If not, copy the URL shown in the terminal and paste it into your browser.

### You are ready to use the application!

Now you should have:
*   Qdrant running in one terminal.
*   FastAPI backend running in another terminal.
*   Streamlit UI running in a third terminal and accessible in your browser.

You can now use the Streamlit interface to:
1.  Enter a **Project Name**.
2.  Select an **LLM Model ID** (or use the default).
3.  Go to the "New Analysis" tab and **upload your CSV file** containing URLs (one URL per line in the first column).
4.  Click "Start Analysis".

Progress will be streamed to the UI (this part is still under development in the UI for full SSE display but the backend supports it).

## Project Structure

(This section would detail the file structure, which you already have visibility of)

## Development Notes

*   **API Documentation:** FastAPI auto-generates OpenAPI documentation, accessible at `/docs` (e.g., `http://localhost:8000/docs`) and ReDoc at `/redoc` when the backend is running.
*   **Stopping the Applications:**
    *   To stop Streamlit or FastAPI: Go to their respective terminals and press `Ctrl+C`.
    *   To stop Qdrant: Go to its terminal and press `Ctrl+C`. You can also manage Docker containers through Docker Desktop.
*   **Troubleshooting:**
    *   **Port Conflicts:** If port `8000`, `8501`, or `6333` are in use, the applications might not start. You may need to stop the other service using that port or configure these applications to use different ports (more advanced).
    *   **API Key Issues:** If Jina or OpenRouter functionalities fail, double-check your API keys in the `.env` file.
    *   **Dependencies:** If you encounter `ModuleNotFoundError`, ensure `poetry install` completed successfully and that you are running commands within the Poetry environment (`poetry shell`).

## Language
All user-facing interface elements, code comments, docstrings, and LLM prompts are in **English**.
