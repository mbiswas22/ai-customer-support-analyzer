# 📊 AI Customer Support Analyzer

A Streamlit web app that analyzes customer support tickets using NLP, interactive clustering, ticket forecasting, and a RAG-powered AI assistant.

## Features

- **Category Classification** — Automatically tags tickets as Billing, Technical, Account, or General
- **Sentiment Analysis** — Detects Positive, Neutral, or Negative tone using TextBlob
- **Urgency Detection** — Flags tickets containing urgent keywords
- **Priority Scoring** — Assigns Low, Medium, or High priority based on urgency, sentiment, and category
- **Interactive Dashboard** — Visualizes ticket trends, category bar chart and sentiment pie chart side by side, plus priority distribution
- **Ticket Forecasting** — Predicts future ticket volume using linear regression; filter by category and choose forecast horizon (1–30 days)
- **Clustering** — Groups tickets into 2–4 clusters using KMeans + PCA; color by cluster, category, sentiment, or priority
- **AI Q&A (RAG)** — Ask natural language questions about your tickets powered by FAISS + Qwen via HuggingFace
- **Branded Header** — Round app icon displayed centered above the title before upload, and inline beside the title after upload
- **Auto-dismiss Notifications** — "Data processed successfully!" message auto-closes after 5 seconds

## Project Structure

```
ai-customer-support-analyzer/
├── app/
│   └── app.py                        # Streamlit UI
├── data/
│   ├── customer_support_tickets.csv  # Primary dataset (4000 rows)
│   ├── customer_support_tickets2.csv # Secondary dataset (2000 rows)
│   └── customer-analysis-icon-1.avif # App icon
├── forecasting/
│   └── forecast_utils.py             # Linear regression forecasting
├── models/
│   ├── model_utils.py                # Classification, sentiment, urgency logic
│   └── cluster_utils.py              # KMeans clustering + PCA
├── rag/
│   └── rag_utils.py                  # Vector store creation and RAG Q&A
├── utils/
│   └── data_utils.py                 # Data loading and feature engineering
├── .env                              # API keys (not committed)
├── requirements.txt
└── README.md
```

## Requirements

- **Python 3.14**

## Setup

1. **Clone the repo and install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your HuggingFace API token**

   Create a `.env` file in the project root:

   ```
   HUGGINGFACEHUB_API_TOKEN=<your_token_here>
   ```

3. **Run the app**
   ```bash
   streamlit run app/app.py
   ```

## Usage

1. The app loads with a centered round icon and title — upload your CSV using the file uploader
2. After upload, the icon moves inline beside the title
3. The app automatically classifies, scores sentiment, detects urgency, and assigns priority
4. A "Data processed successfully!" message appears for 5 seconds then automatically dismisses
5. Use the **Dashboard** tab to explore visual summaries — Category Distribution (bar chart) and Sentiment Distribution (pie chart) are shown side by side
6. Use the **Data** tab to filter and browse tickets
7. Use the **AI Insights** tab to ask questions like _"What are the most common issues?"_
8. Use the **Forecast** tab to predict ticket volume — select a category and drag the days slider
9. Use the **Clusters** tab to explore ticket groupings — adjust cluster count (2–4) and color dimension interactively
10. Use the **Priority filter** in the sidebar to narrow tickets by Low, Medium, or High priority — and click **Send Alert for High Priority Tickets** (also in the sidebar) to email the top 10 most critical high priority tickets ranked by a severity score based on keyword weight, sentiment, response time, and text length

## CSV Format

Your input CSV should include at minimum:

| column                  | description                                        |
| ----------------------- | -------------------------------------------------- |
| `text`                  | The ticket message content                         |
| `created_at`            | Ticket creation date/timestamp                     |
| `response_time_hours`   | Hours taken to respond (used in clustering)        |
| `priority` _(optional)_ | Low / Medium / High — auto-assigned if not present |

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [TextBlob](https://textblob.readthedocs.io/) — Sentiment analysis
- [scikit-learn](https://scikit-learn.org/) — KMeans clustering, PCA, linear regression forecasting
- [Plotly](https://plotly.com/python/) — Interactive clustering scatter chart and sentiment pie chart
- [LangChain](https://www.langchain.com/) + [FAISS](https://github.com/facebookresearch/faiss) — RAG pipeline
- [HuggingFace](https://huggingface.co/) — Embeddings (`all-MiniLM-L6-v2`) and LLM (`Qwen2.5-72B-Instruct`)
- [Pandas](https://pandas.pydata.org/) — Data processing
- [Pillow](https://python-pillow.org/) — App icon rendering
