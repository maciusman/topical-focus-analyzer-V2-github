# 🌐 Topical Focus Analyzer - User Guide

---

### 💡 Understanding Site Focus Score & Site Radius Score

Before using the tool, understand its core metrics:

*   **Site Focus Score (0-100):** Measures the thematic similarity of pages across the site. A **higher score** indicates strong thematic cohesion and specialization.
*   **Site Radius Score (0-100):** Reflects how tightly pages cluster around the site's central theme. A **higher score** means pages don't deviate much from the overall theme (smaller "radius").

**Why check this?** High scores can signal topical authority to search engines, help refine content strategy, and improve user experience.

---

## 1. 🚀 Introduction

The Topical Focus Analyzer helps understand a website's structure and content themes using sitemaps and optional page content analysis. It calculates the **Site Focus Score** and **Site Radius Score**, visualizes content relationships, and identifies potential content cannibalization.

This guide covers setting up and running the application based on the `multi_sitemap_app.py` version.

## 2. 📋 Prerequisites

*   **Python:** Version 3.9+ installed ([python.org](https://www.python.org/)).
*   **Google AI API Key (Optional):** Needed for the AI Summary feature ([Google AI Studio](https://aistudio.google.com/app/apikey)).

## 3. 🛠️ Installation & Setup

1.  **Create Project Directory & Navigate Into It:**
    ```bash
    mkdir topical-focus-analyzer
    cd topical-focus-analyzer
    ```

2.  **Create & Activate Virtual Environment:**
    ```bash
    # Create
    python -m venv venv

    # Activate (Windows - Command Prompt/PowerShell)
    venv\Scripts\activate

    # Activate (macOS/Linux - Bash/Zsh)
    source venv/bin/activate
    ```
    *(You should see `(venv)` in your prompt)*

3.  **Create `requirements.txt` File:**
    Create a file named `requirements.txt` in the `topical-focus-analyzer` directory with the following content:
    ```text
    # Core Libraries
    requests
    beautifulsoup4
    lxml
    pandas
    numpy==1.26.4
    scikit-learn==1.4.2
    plotly
    streamlit
    python-dotenv

    # AI Summarization
    google-generativeai

    # Content Extraction (Simplified)
    trafilatura
    regex
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `trafilatura` fails, the app will use a fallback for content extraction).*

5.  **Create Project Files & Structure:**
    Ensure the following files and directory structure exist within `topical-focus-analyzer`. You will need to populate these with the Python code developed previously.
    ```
    topical-focus-analyzer/
    ├── modules/
    │   ├── __init__.py
    │   ├── analyzer.py
    │   ├── content_extractor.py
    │   ├── dimensionality_reducer.py
    │   ├── llm_summarizer.py
    │   ├── simple_vectorizer.py
    │   ├── sitemap_finder.py
    │   └── sitemap_parser.py
    ├── .env
    ├── multi_sitemap_app.py
    └── requirements.txt
    ```
    *(Use `mkdir modules` and `touch <filename>` (macOS/Linux) or `New-Item -ItemType File -Path "<filename>"` (Windows PowerShell) to create empty files if needed).*

## 4. ⚙️ Configuration

1.  Open the `.env` file.
2.  Add your Google AI API Key (required only for AI Summary):
    ```dotenv
    GOOGLE_API_KEY=your_google_api_key_here
    ```
3.  Save the file.

## 5. ▶️ Running the Application

1.  Ensure your virtual environment (`venv`) is active.
2.  Run the Streamlit app from the `topical-focus-analyzer` directory:
    ```bash
    streamlit run multi_sitemap_app.py
    ```
3.  Access the app in your web browser, usually at `http://localhost:8501`.

## 6. 🖱️ Basic Usage Workflow

1.  **Enter Domain** in the sidebar.
2.  Click **Find Sitemaps**.
3.  **Select Sitemaps** to analyze using the checkboxes.
4.  Configure optional **URL Filters** and **Analysis Options** (like enabling "Analyze Page Content").
5.  Optionally enable **Generate AI Summary** (requires API key in `.env`).
6.  Click **Process Selected Sitemaps**.
7.  Explore results in the main area tabs: **Overview**, **URL Details**, **Visual Map**, **Cannibalization**, **Content Inspector**, **Processing Log**.

---

### Understanding Advanced Analysis Options

These optional settings allow fine-tuning:

*   **t-SNE Perplexity (Default: 15):** Influences the visual map layout. Lower values emphasize local clusters, higher values global structure. Adjust based on dataset size (lower for small, higher for large).
*   **Site Focus Score Scaling (k1) (Default: 5.0):** Adjusts the sensitivity of the Site Focus Score. Higher values make the score more sensitive to similarity variations. Adjust if scores seem consistently too high or low.
*   **Site Radius Score Scaling (k2) (Default: 5.0):** Adjusts the sensitivity of the Site Radius Score. Higher values make the score more sensitive to outliers and content spread. Adjust if scores seem counter-intuitive for site focus.

---

## 7. 🧩 Main Files Overview

*   `multi_sitemap_app.py`: Main Streamlit application interface and orchestrator.
*   `.env`: Stores the Google API key.
*   `requirements.txt`: Lists Python dependencies.
*   `modules/`: Contains core logic:
    *   `sitemap_finder.py`: Discovers sitemaps.
    *   `sitemap_parser.py`: Parses sitemaps and extracts URLs.
    *   `content_extractor.py`: Fetches and extracts main text content from pages (optional).
    *   `simple_vectorizer.py`: Converts text (URL paths/content) into numerical TF-IDF vectors.
    *   `dimensionality_reducer.py`: Reduces vector dimensions using t-SNE for visualization, calculates centroid.
    *   `analyzer.py`: Calculates Site Focus Score, Site Radius Score, and finds potential duplicates.
    *   `llm_summarizer.py`: Generates AI summary using Google Gemini (optional).

---
