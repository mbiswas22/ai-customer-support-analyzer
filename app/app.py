import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from utils.data_utils import load_data, add_features, filter_data
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
    df["created_at"] = pd.to_datetime(df["created_at"])

    st.success("Data processed successfully!")

    # ---------------------------
    # GLOBAL FILTERS (sidebar)
    # ---------------------------
    st.sidebar.header("🔍 Filter Tickets")

    category_filter = st.sidebar.selectbox(
        "Category",
        ["All"] + sorted(df["category"].unique().tolist())
    )

    sentiment_filter = st.sidebar.selectbox(
        "Sentiment",
        ["All"] + sorted(df["sentiment"].unique().tolist())
    )

    urgent_filter = st.sidebar.checkbox("Show Urgent Tickets Only")

    st.sidebar.markdown("**Date Range**")
    start_date = st.sidebar.date_input("From", value=df["created_at"].min())
    end_date = st.sidebar.date_input("To", value=df["created_at"].max())

    filtered_df = filter_data(
        df,
        category=category_filter,
        sentiment=sentiment_filter,
        urgent_only=urgent_filter,
        start_date=start_date,
        end_date=end_date
    )

    st.sidebar.markdown(f"**Showing {len(filtered_df)} of {len(df)} tickets**")

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

        col1.metric("Total Tickets", len(filtered_df))
        col2.metric("Urgent Tickets", filtered_df["urgent"].sum())
        col3.metric("Negative Sentiment", (filtered_df["sentiment"] == "Negative").sum())

        st.subheader("Category Distribution")
        st.bar_chart(filtered_df["category"].value_counts())

        st.subheader("Sentiment Distribution")
        st.bar_chart(filtered_df["sentiment"].value_counts())

        st.subheader("Tickets Over Time")
        filtered_df["created_at"] = pd.to_datetime(df["created_at"])
        trend = filtered_df.groupby("created_at").size()
        st.line_chart(trend)

    # ---------------------------
    # TAB 2: DATA TABLE
    # ---------------------------
    with tab2:
        st.subheader("Filtered Data")
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