# 📊 AI Customer Support Analyzer

A Streamlit web app that analyzes customer support tickets using NLP and a RAG-powered AI assistant.

## Features

- **Category Classification** — Automatically tags tickets as Billing, Technical, Account, or General
- **Sentiment Analysis** — Detects Positive, Neutral, or Negative tone using TextBlob
- **Urgency Detection** — Flags tickets containing urgent keywords
- **Interactive Dashboard** — Visualizes ticket trends, category and sentiment distributions
- **AI Q&A (RAG)** — Ask natural language questions about your tickets powered by FAISS + Qwen via HuggingFace

## Project Structure

```
ai-customer-support-analyzer/
├── app/
│   └── app.py              # Streamlit UI
├── data/
│   └── customer_support_tickets.csv
├── models/
│   └── model_utils.py      # Classification, sentiment, urgency logic
├── rag/
│   └── rag_utils.py        # Vector store creation and RAG Q&A
├── utils/
│   └── data_utils.py       # Data loading and feature engineering
├── .env                    # API keys (not committed)
├── requirements.txt
└── README.md
```

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

1. Upload a CSV file containing a `text` column with support ticket messages and a `created_at` date column
2. The app will automatically classify, score sentiment, and flag urgent tickets
3. Use the **Dashboard** tab to explore visual summaries
4. Use the **Data** tab to filter and browse tickets
5. Use the **AI Insights** tab to ask questions like *"What are the most common issues?"*

## CSV Format

Your input CSV should include at minimum:

| column      | description                        |
|-------------|------------------------------------|
| `text`      | The ticket message content         |
| `created_at`| Ticket creation date/timestamp     |

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [TextBlob](https://textblob.readthedocs.io/) — Sentiment analysis
- [LangChain](https://www.langchain.com/) + [FAISS](https://github.com/facebookresearch/faiss) — RAG pipeline
- [HuggingFace](https://huggingface.co/) — Embeddings (`all-MiniLM-L6-v2`) and LLM (`Qwen2.5-72B-Instruct`)
- [Pandas](https://pandas.pydata.org/) — Data processing
