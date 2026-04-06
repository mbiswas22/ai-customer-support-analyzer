import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from utils.data_utils import load_data, add_features
from models.model_utils import process_predictions
from rag.rag_utils import create_vector_store, answer_question

st.set_page_config(page_title="Support Analyzer", layout="wide")

st.title("📊 AI Customer Support Analyzer")

# Upload file
uploaded_file = st.file_uploader("Upload Support Tickets CSV")

if uploaded_file:

    # Load data
    df = load_data(uploaded_file)
    df = add_features(df)
    df = process_predictions(df)

    st.success("Data processed successfully!")

    # ---------------------------
    # TABS
    # ---------------------------
    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard",
        "📋 Data",
        "🤖 AI Insights"
    ])

    # ---------------------------
    # TAB 1: DASHBOARD
    # ---------------------------
    with tab1:
        st.subheader("Ticket Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Tickets", len(df))
        col2.metric("Urgent Tickets", df["urgent"].sum())
        col3.metric("Negative Sentiment", (df["sentiment"] == "Negative").sum())

        st.subheader("Category Distribution")
        st.bar_chart(df["category"].value_counts())

        st.subheader("Sentiment Distribution")
        st.bar_chart(df["sentiment"].value_counts())

        st.subheader("Tickets Over Time")
        df["created_at"] = pd.to_datetime(df["created_at"])
        trend = df.groupby("created_at").size()
        st.line_chart(trend)

    # ---------------------------
    # TAB 2: DATA TABLE
    # ---------------------------
    with tab2:
        st.subheader("Filtered Data")

        category_filter = st.selectbox(
            "Filter by Category",
            ["All"] + list(df["category"].unique())
        )

        if category_filter != "All":
            filtered_df = df[df["category"] == category_filter]
        else:
            filtered_df = df

        st.dataframe(filtered_df)

    # ---------------------------
    # TAB 3: AI Q&A
    # ---------------------------
    with tab3:
        st.subheader("Ask Questions About Tickets")

        vector_db = create_vector_store(df["text"].tolist())

        query = st.text_input("Ask something like: What are common issues?")

        if query:
            answer = answer_question(vector_db, query)
            st.info(answer)