# AI Customer Support Analyzer — Project Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [Tech Stack](#3-tech-stack)
4. [System Requirements](#4-system-requirements)
5. [Project Structure](#5-project-structure)
6. [Dataset](#6-dataset)
7. [Module Breakdown](#7-module-breakdown)
   - 7.1 [app.py — UI Orchestrator](#71-apppy--ui-orchestrator)
   - 7.2 [data_utils.py — Data Layer](#72-data_utilspy--data-layer)
   - 7.3 [model_utils.py — NLP & Classification](#73-model_utilspy--nlp--classification)
   - 7.4 [cluster_utils.py — Clustering Engine](#74-cluster_utilspy--clustering-engine)
   - 7.5 [forecast_utils.py — Forecasting Engine](#75-forecast_utilspy--forecasting-engine)
   - 7.6 [rag_utils.py — RAG AI Assistant](#76-rag_utilspy--rag-ai-assistant)
8. [Feature Deep Dive](#8-feature-deep-dive)
   - 8.1 [Branded Header](#81-branded-header)
   - 8.2 [Dashboard Tab](#82-dashboard-tab)
   - 8.3 [Data Tab](#83-data-tab)
   - 8.4 [AI Insights Tab](#84-ai-insights-tab)
   - 8.5 [Forecast Tab](#85-forecast-tab)
   - 8.6 [Clusters Tab](#86-clusters-tab)
9. [Data Pipeline](#9-data-pipeline)
10. [Priority Scoring Logic](#10-priority-scoring-logic)
11. [System Design & Flow](#11-system-design--flow)
12. [Setup & Installation](#12-setup--installation)
13. [Environment Configuration](#13-environment-configuration)
14. [Running the App](#14-running-the-app)
15. [CSV Input Format](#15-csv-input-format)
16. [Dependencies](#16-dependencies)

---

## 1. Project Overview

The **AI Customer Support Analyzer** is a fully interactive, browser-based web application built with Streamlit. It enables support teams and analysts to upload customer support ticket data in CSV format and instantly gain deep insights through automated NLP processing, visual analytics, machine learning clustering, time-series forecasting, and a conversational AI assistant.

The application removes the need for manual ticket triage by automatically classifying every ticket by category, detecting sentiment, flagging urgency, and assigning a priority level. All results are presented through an intuitive multi-tab dashboard with interactive charts and filters.

---

## 2. Objectives

| Objective | How It Is Achieved |
|---|---|
| Automate ticket categorization | Keyword-based classifier in `model_utils.py` |
| Understand customer sentiment | TextBlob polarity scoring |
| Identify urgent tickets instantly | Keyword urgency detection |
| Prioritize workload | Rule-based priority scoring (high / medium / low) |
| Visualize trends and distributions | Streamlit bar charts and line charts |
| Predict future ticket volume | Linear regression on daily ticket counts |
| Discover hidden ticket patterns | KMeans clustering + PCA visualization |
| Enable natural language Q&A | RAG pipeline with FAISS + Qwen LLM |
| Provide a polished branded UI | Custom HTML/CSS header with round icon |

---

## 3. Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| UI Framework | Streamlit | 1.45.1 | Web app rendering and interactivity |
| Data Processing | Pandas | 3.0.1 | DataFrame operations, filtering, aggregation |
| Numerical Computing | NumPy | 2.4.3 | Array operations, forecast math |
| NLP | TextBlob | 0.19.0 | Sentiment polarity scoring |
| Machine Learning | scikit-learn | 1.6.1 | KMeans, PCA, LinearRegression, LabelEncoder, StandardScaler |
| Visualization | Plotly | ≥5.0.0 | Interactive clustering scatter chart |
| Vector Store | FAISS (CPU) | 1.11.0 | Similarity search for RAG retrieval |
| Embeddings | sentence-transformers | 4.1.0 | all-MiniLM-L6-v2 text embeddings |
| LLM Orchestration | LangChain + LangChain-Community | 0.3.25 / 0.3.24 | RAG chain construction |
| LLM Integration | LangChain-HuggingFace | 0.1.2 | HuggingFace model binding |
| LLM Inference | HuggingFace Hub | 0.32.3 | Qwen2.5-72B-Instruct API calls |
| Image Processing | Pillow | bundled | AVIF icon loading and JPEG conversion |
| Secrets Management | python-dotenv | 1.1.0 | Loading API keys from .env file |
| Deep Learning Runtime | torchvision | 0.22.0 | Required by sentence-transformers |

---

## 4. System Requirements

- **Python**: 3.14
- **Operating System**: Windows / macOS / Linux
- **Internet**: Required for HuggingFace Inference API (AI Insights tab only)
- **HuggingFace Account**: Free account with API token for the RAG Q&A feature
- **RAM**: Minimum 4 GB recommended (FAISS vector store is in-memory)

---

## 5. Project Structure

```
ai-customer-support-analyzer/
│
├── app/
│   └── app.py                         # Main Streamlit application — UI orchestrator
│
├── data/
│   ├── customer_support_tickets.csv   # Primary dataset — 4000 rows
│   ├── customer_support_tickets2.csv  # Secondary dataset — 2000 rows
│   └── customer-analysis-icon-1.avif  # Branded app icon (AVIF format)
│
├── forecasting/
│   └── forecast_utils.py              # Daily series builder + LinearRegression forecaster
│
├── models/
│   ├── model_utils.py                 # Category classifier, sentiment analyzer, urgency detector
│   └── cluster_utils.py               # Feature encoder, KMeans clustering, PCA reducer
│
├── rag/
│   └── rag_utils.py                   # FAISS vector store builder + RAG Q&A pipeline
│
├── utils/
│   └── data_utils.py                  # CSV loader, feature engineering, sidebar filtering
│
├── .env                               # API keys — NOT committed to version control
├── requirements.txt                   # All Python dependencies with pinned versions
└── README.md                          # Project readme
```

Each directory has a single, focused responsibility. No module imports from another module at the same level — all cross-module calls are orchestrated exclusively through `app.py`.

---

## 6. Dataset

### Primary Dataset — `customer_support_tickets.csv`

| Property | Value |
|---|---|
| Total rows | 4,000 |
| Date range | 2024-01-01 to 2025-04-04 |
| Columns | 8 |

The dataset was built in two phases:
- **Phase 1 (500 rows)**: Manually seeded data covering January–March 2024
- **Phase 2 (3,500 rows)**: Programmatically generated with realistic daily volume variation — higher on weekdays, gradual upward trend over time simulating business growth

### Secondary Dataset — `customer_support_tickets2.csv`

| Property | Value |
|---|---|
| Total rows | 2,000 |
| Date range | 2024-01-01 to 2025-07-14 (random sampling) |
| Columns | 8 (identical schema) |

Generated independently with a different random seed to provide a distinct dataset for testing and comparison.

### Column Schema

| Column | Type | Description |
|---|---|---|
| `ticket_id` | Integer | Unique ticket identifier |
| `text` | String | The raw ticket message content |
| `category` | String | Billing / Technical / Account / General |
| `sentiment` | String | Positive / Neutral / Negative |
| `urgent` | Boolean | True if ticket contains urgent keywords |
| `created_at` | Date | Ticket creation date (YYYY-MM-DD) |
| `response_time_hours` | Integer | Hours taken to respond (1–72) |
| `priority` | String | high / medium / low |

### Ticket Text Categories

| Category | Example Texts |
|---|---|
| Billing | Billing invoice incorrect, Charged twice for subscription, Refund not processed, Payment failed again, Unable to update payment method |
| Technical | App crashes on login, Website not loading, Error while uploading file, Feature not working properly, System timeout issue |
| Account | Unable to reset password, Login issues with account, Account locked unexpectedly, Profile update failed, Verification email not received |
| General | General inquiry, Request for demo, Feedback on service, How to use this feature?, Need more information about product |

---

## 7. Module Breakdown

### 7.1 `app.py` — UI Orchestrator

**Location**: `app/app.py`

This is the entry point of the application. It is responsible for:

- Configuring the Streamlit page layout (`wide` mode)
- Loading and base64-encoding the app icon from the `data/` folder
- Rendering the branded header (centered before upload, inline after upload) using `st.empty()` placeholder
- Accepting the CSV file upload via `st.file_uploader`
- Calling the data pipeline in sequence: `load_data` → `add_features` → `process_predictions` → priority assignment
- Rendering the sidebar with global filters (category, sentiment, urgency, date range)
- Passing `filtered_df` to each of the 5 tabs
- Orchestrating all module calls for Dashboard, Data, AI Insights, Forecast, and Clusters tabs

**Key design decision**: `st.empty()` is used to reserve the title slot above the file uploader widget. This allows the title to always appear visually above the uploader while Streamlit's top-down rendering model is respected. After upload, the same slot is updated with the inline icon layout.

---

### 7.2 `data_utils.py` — Data Layer

**Location**: `utils/data_utils.py`

**Functions**:

#### `load_data(file)`
- Reads the uploaded CSV into a Pandas DataFrame
- Drops all rows with any null values (`dropna`)
- Returns the clean DataFrame

#### `add_features(df)`
- Adds `text_length` — character count of each ticket text
- Adds `word_count` — word count of each ticket text
- These engineered features are used as inputs to the clustering model

#### `filter_data(df, category, sentiment, urgent_only, start_date, end_date)`
- Applies up to 5 simultaneous filters on a copy of the DataFrame
- Category filter: exact match against the `category` column
- Sentiment filter: exact match against the `sentiment` column
- Urgent filter: boolean mask on the `urgent` column
- Date range: inclusive filter on `created_at` using `pd.to_datetime` conversion
- Returns the filtered DataFrame without modifying the original

---

### 7.3 `model_utils.py` — NLP & Classification

**Location**: `models/model_utils.py`

**Functions**:

#### `classify(text)`
Keyword-based rule classifier:
- Contains "payment" or "refund" → **Billing**
- Contains "error" or "crash" → **Technical**
- Contains "account" or "login" → **Account**
- Otherwise → **General**

#### `get_sentiment(text)`
Uses TextBlob's `sentiment.polarity` score:
- Score < 0 → **Negative**
- Score == 0 → **Neutral**
- Score > 0 → **Positive**

TextBlob computes polarity using a pattern-based lexicon, assigning scores between -1.0 (most negative) and +1.0 (most positive).

#### `is_urgent(text)`
Checks for presence of any of these keywords (case-insensitive):
- `urgent`, `asap`, `immediately`, `not working`
- Returns `True` if any keyword is found, `False` otherwise

#### `process_predictions(df)`
Applies all three functions above to every row in the DataFrame using `.apply()`, adding `category`, `sentiment`, and `urgent` columns. This is the main entry point called from `app.py`.

---

### 7.4 `cluster_utils.py` — Clustering Engine

**Location**: `models/cluster_utils.py`

**Functions**:

#### `build_cluster_features(df)`
Constructs a 7-dimensional numerical feature matrix from the DataFrame:

| Feature | Type | Encoding |
|---|---|---|
| `category` | Categorical | LabelEncoder (integer) |
| `sentiment` | Categorical | LabelEncoder (integer) |
| `priority` | Categorical | LabelEncoder (integer) |
| `urgent` | Boolean | Cast to int (0/1) |
| `response_time_hours` | Numeric | As-is |
| `text_length` | Numeric | As-is |
| `word_count` | Numeric | As-is |

All 7 features are then normalized using `StandardScaler` (zero mean, unit variance) to prevent high-magnitude numeric features from dominating the clustering.

#### `run_clustering(df, n_clusters=3)`
1. Calls `build_cluster_features` to get the scaled feature matrix `X`
2. Runs `KMeans(n_clusters, random_state=42, n_init=10)` to assign cluster labels
3. Runs `PCA(n_components=2)` to reduce the 7D feature space to 2D for visualization
4. Returns a DataFrame with original ticket columns plus `cluster`, `pca_x`, and `pca_y` columns ready for Plotly

**Why PCA?** The 7-dimensional feature space cannot be directly visualized. PCA finds the two directions of maximum variance, projecting each ticket to a 2D point that preserves as much structure as possible.

---

### 7.5 `forecast_utils.py` — Forecasting Engine

**Location**: `forecasting/forecast_utils.py`

**Functions**:

#### `build_daily_series(df, category=None)`
- Optionally filters the DataFrame to a single category
- Groups tickets by date using `.groupby("date").size()`
- Returns a sorted time series DataFrame with columns `date` and `count`

#### `forecast_tickets(df, days_ahead=7, category=None)`
1. Calls `build_daily_series` to get historical daily counts
2. Returns `(None, None)` if fewer than 3 data points exist
3. Converts dates to integer ordinals (days since a fixed epoch) as the feature `X`
4. Fits `LinearRegression` on `X` (ordinals) → `y` (daily counts)
5. Generates `days_ahead` future dates starting from the day after the last known date
6. Predicts ticket counts for future dates, clips negatives to 0, rounds to integers
7. Returns the historical series and a forecast DataFrame with `date` and `predicted_tickets`

**Why linear regression?** It is interpretable, fast, requires no hyperparameter tuning, and works well for capturing the overall trend in ticket volume over time. The date ordinal is a monotonically increasing integer, making it a natural single feature for trend modeling.

---

### 7.6 `rag_utils.py` — RAG AI Assistant

**Location**: `rag/rag_utils.py`

Implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to ask natural language questions about their ticket data.

**Functions**:

#### `create_vector_store(texts)`
1. Loads the `all-MiniLM-L6-v2` sentence embedding model via `HuggingFaceEmbeddings`
2. Encodes all ticket texts into dense vector embeddings
3. Builds a FAISS (Facebook AI Similarity Search) in-memory index from those embeddings
4. Returns the vector store object

`all-MiniLM-L6-v2` is a lightweight but highly capable sentence transformer model that maps text to 384-dimensional vectors optimized for semantic similarity.

#### `answer_question(vector_db, query)`
1. Creates a retriever from the FAISS vector store
2. Retrieves the most semantically similar ticket texts to the user's query
3. Concatenates retrieved texts into a context string
4. Constructs a prompt: *"You are a customer support analyst. Use the context below to answer the question..."*
5. Sends the prompt to `Qwen/Qwen2.5-72B-Instruct` via `HuggingFace InferenceClient`
6. Returns the model's text response (max 300 tokens)

**RAG Architecture**:
```
User Query
    │
    ▼
FAISS Retriever ──► Top-K similar ticket texts
    │
    ▼
Prompt Builder ──► "Context: [tickets] \n Question: [query]"
    │
    ▼
Qwen2.5-72B-Instruct (HuggingFace API)
    │
    ▼
Natural Language Answer
```

---

## 8. Feature Deep Dive

### 8.1 Branded Header

The app icon (`customer-analysis-icon-1.avif`) is loaded once at startup using Pillow, converted to JPEG in-memory, and base64-encoded into a data URI. This avoids any file path issues at runtime and embeds the image directly in the HTML.

Two HTML layouts are pre-built as string constants:

- **`_ICON_HTML_CENTER`**: Flexbox column layout — icon centered above the title, both horizontally centered on the page. Shown before a file is uploaded.
- **`_ICON_HTML_INLINE`**: Flexbox row layout — 64px round icon to the left of the title text. Shown after a file is uploaded.

`st.empty()` reserves a slot above the file uploader. After Streamlit evaluates whether a file was uploaded, the correct HTML is injected into that slot, making the title always appear above the upload widget regardless of which layout is active.

---

### 8.2 Dashboard Tab

Displays a high-level overview of the **filtered** dataset:

- **Metrics row**: Total Tickets, Urgent Tickets count, Negative Sentiment count
- **Category Distribution**: Horizontal bar chart of ticket counts per category
- **Sentiment Distribution**: Bar chart of Positive / Neutral / Negative counts
- **Tickets Over Time**: Line chart of daily ticket volume, showing trends across the date range

All charts respond instantly to sidebar filter changes.

---

### 8.3 Data Tab

Renders the full filtered DataFrame as an interactive `st.dataframe` table. Users can:
- Sort by any column by clicking the column header
- Scroll horizontally and vertically
- See all enriched columns including category, sentiment, urgent, priority, text_length, word_count

---

### 8.4 AI Insights Tab

Provides a natural language interface to the ticket data:

1. On tab load, `create_vector_store` builds a FAISS index from all ticket texts (uses the full unfiltered `df` for maximum context)
2. User types a question in the text input
3. The RAG pipeline retrieves relevant tickets and sends them with the question to Qwen2.5-72B
4. The answer is displayed in a styled info box

Example questions:
- *"What are the most common issues?"*
- *"Which category has the most negative sentiment?"*
- *"Are there any billing problems reported recently?"*

---

### 8.5 Forecast Tab

Predicts future daily ticket volume using historical trends:

**Controls**:
- **Category dropdown**: Filter forecast to a specific category (Billing, Technical, Account, General) or All
- **Days slider**: Choose forecast horizon from 1 to 30 days

**Output**:
- Combined line chart with two series — Historical (solid) and Forecast (continuation)
- Metric card showing predicted ticket count for tomorrow specifically
- Detailed forecast table with date-by-date predictions

The model is re-fitted live whenever the category or days slider changes, giving instant interactive feedback.

---

### 8.6 Clusters Tab

Groups tickets into 2–4 clusters using unsupervised machine learning:

**Controls**:
- **Cluster slider (2–4)**: Re-runs KMeans with the selected number of clusters
- **Color by dropdown**: Switch point colors between cluster label, category, sentiment, or priority

**Output**:
- Interactive Plotly scatter chart — each point is a ticket projected to 2D via PCA. Hovering shows the ticket text, category, sentiment, priority, urgency, and response time.
- Cluster summary table — for each cluster: ticket count, dominant category, dominant sentiment, dominant priority, average response time

**Use cases**:
- Identify a cluster of high-urgency billing tickets needing immediate attention
- Discover a cluster of positive general inquiries that could be handled by a chatbot
- Spot outlier tickets that don't fit typical patterns

---

## 9. Data Pipeline

Every time a CSV is uploaded, the following pipeline runs in sequence:

```
Step 1 — load_data()
  Read CSV → Drop null rows → Raw DataFrame

Step 2 — add_features()
  + text_length (character count)
  + word_count

Step 3 — process_predictions()
  + category  (keyword classifier)
  + sentiment (TextBlob polarity)
  + urgent    (keyword detection)

Step 4 — Priority Assignment
  If 'priority' column exists in CSV → use as-is
  Else → apply rule-based scoring:
    urgent=True AND sentiment=Negative  → high
    urgent=True OR sentiment=Negative
      OR category in (Billing, Technical) → medium
    Otherwise                            → low

Step 5 — created_at conversion
  pd.to_datetime() for date operations

Step 6 — Sidebar Filtering
  filter_data() → filtered_df (used by all tabs)
```

The full enriched DataFrame is passed to the RAG and Forecast modules. The filtered DataFrame is passed to Dashboard, Data, and Clusters.

---

## 10. Priority Scoring Logic

Priority is determined by a three-tier rule evaluated in order:

| Condition | Priority |
|---|---|
| `urgent == True` AND `sentiment == "Negative"` | **high** |
| `urgent == True` OR `sentiment == "Negative"` OR `category in ("Billing", "Technical")` | **medium** |
| None of the above | **low** |

This logic reflects real-world support triage: a ticket that is both urgent and negative needs immediate attention (high), while billing and technical issues are inherently more impactful than general inquiries even without explicit urgency signals (medium).

**Distribution in primary dataset (4000 rows)**:
- medium: ~2,577 (64%)
- low: ~1,193 (30%)
- high: ~230 (6%)

---

## 11. System Design & Flow

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (app.py)                    │
│                                                             │
│  Branded Header (st.empty placeholder)                      │
│  File Uploader                                              │
│  Sidebar Filters                                            │
│                                                             │
│  ┌──────────┬─────────┬───────────┬──────────┬──────────┐  │
│  │Dashboard │  Data   │AI Insights│ Forecast │ Clusters │  │
│  └──────────┴─────────┴───────────┴──────────┴──────────┘  │
└─────────────────────────────────────────────────────────────┘
       │           │            │           │           │
       ▼           ▼            ▼           ▼           ▼
  data_utils  data_utils   rag_utils  forecast_utils cluster_utils
  model_utils              (FAISS +
                            Qwen API)
```

### End-to-End Flow

```
User uploads CSV
      │
      ▼
load_data → add_features → process_predictions → priority assign
      │
      ▼
Enriched DataFrame
      │
      ├──► Sidebar filters ──► filtered_df
      │                              │
      │              ┌───────────────┼──────────────────┐
      │              ▼               ▼                  ▼
      │          Dashboard         Data Tab          Clusters
      │         (charts)         (dataframe)     (KMeans+PCA+Plotly)
      │
      ├──► Full df ──► RAG vector store ──► FAISS ──► Qwen LLM ──► Answer
      │
      └──► Full df ──► Daily series ──► LinearRegression ──► Forecast chart
```

### External Dependencies

Only one external network call is made at runtime — the HuggingFace Inference API in the AI Insights tab. All other processing (NLP, clustering, forecasting, filtering) runs entirely locally.

---

## 12. Setup & Installation

### Step 1 — Clone the repository

```bash
git clone <repository-url>
cd ai-customer-support-analyzer
```

### Step 2 — Ensure Python 3.14 is installed

```bash
python --version
# Should output: Python 3.14.x
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure environment variables

Create a `.env` file in the project root:

```
HUGGINGFACEHUB_API_TOKEN=<your_huggingface_token>
```

To get a free HuggingFace API token:
1. Create an account at https://huggingface.co
2. Go to Settings → Access Tokens
3. Create a new token with Read permissions
4. Paste it into the `.env` file

### Step 5 — Run the application

```bash
streamlit run app/app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

---

## 13. Environment Configuration

The `.env` file must be placed in the project root (same level as `requirements.txt`). It is loaded automatically by `python-dotenv` when `rag_utils.py` is imported.

```
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Important**: Never commit the `.env` file to version control. It is listed in `.gitignore` by convention.

The token is only used in the AI Insights tab. All other features (Dashboard, Data, Forecast, Clusters) work fully offline without any API key.

---

## 14. Running the App

```bash
streamlit run app/app.py
```

**Usage flow**:

1. App loads — centered round icon above the title is displayed
2. Upload a CSV file using the file uploader (use `customer_support_tickets.csv` or `customer_support_tickets2.csv` from the `data/` folder)
3. Icon moves inline beside the title after upload
4. "Data processed successfully!" confirmation appears
5. Use the sidebar to filter by category, sentiment, urgency, and date range
6. Navigate between the 5 tabs to explore different views
7. In the Forecast tab — select a category and drag the days slider to see predictions update live
8. In the Clusters tab — adjust the cluster count and color dimension to explore groupings
9. In the AI Insights tab — type any natural language question about the tickets

---

## 15. CSV Input Format

The application expects a CSV with the following columns:

| Column | Required | Type | Description |
|---|---|---|---|
| `text` | Yes | String | The raw ticket message |
| `created_at` | Yes | Date string | Ticket creation date (YYYY-MM-DD) |
| `response_time_hours` | Yes | Integer | Hours to first response (used in clustering) |
| `category` | No | String | Auto-assigned if missing |
| `sentiment` | No | String | Auto-assigned if missing |
| `urgent` | No | Boolean | Auto-assigned if missing |
| `priority` | No | String | Auto-assigned if missing |
| `ticket_id` | No | Integer | Optional identifier |

If `category`, `sentiment`, `urgent`, or `priority` columns are present in the CSV, they are overwritten by the NLP pipeline (except `priority` which is preserved if already present).

---

## 16. Dependencies

```
pandas==3.0.1               # DataFrame operations
numpy==2.4.3                # Numerical computing
scikit-learn==1.6.1         # KMeans, PCA, LinearRegression, preprocessing
streamlit==1.45.1           # Web UI framework
langchain==0.3.25           # RAG chain orchestration
langchain-community==0.3.24 # FAISS vector store integration
langchain-huggingface==0.1.2# HuggingFace model binding for LangChain
faiss-cpu==1.11.0           # In-memory vector similarity search
sentence-transformers==4.1.0# all-MiniLM-L6-v2 text embeddings
textblob==0.19.0            # Sentiment polarity analysis
torchvision==0.22.0         # Required runtime for sentence-transformers
huggingface-hub==0.32.3     # HuggingFace Inference API client
python-dotenv==1.1.0        # .env file loading
plotly>=5.0.0               # Interactive scatter chart for clustering
```

All CPU-only dependencies — no GPU required.

---

*Document version: 1.0 | Project: AI Customer Support Analyzer*
