import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import base64, io, time
from PIL import Image

from utils.data_utils import load_data, add_features, filter_data
from models.model_utils import process_predictions
from utils.alert_utils import send_high_priority_alert, generate_alerts
from rag.rag_utils import create_vector_store, answer_question
from forecasting.forecast_utils import forecast_tickets
from models.cluster_utils import run_clustering
import plotly.express as px

st.set_page_config(page_title="Support Analyzer", layout="wide")

# Centered round icon above title
_img = Image.open(os.path.join(os.path.dirname(__file__), "..", "data", "icon.jpeg")).convert("RGB")
_buf = io.BytesIO()
_img.save(_buf, format="JPEG")
_b64 = base64.b64encode(_buf.getvalue()).decode()

# Before upload: centered icon above centered title
_ICON_HTML_CENTER = f"""
<div style="display:flex; flex-direction:column; align-items:center; margin-bottom:12px;">
    <img src="data:image/jpeg;base64,{_b64}"
         style="width:250px; height:250px; border-radius:32%; object-fit:cover;
                box-shadow:0 4px 14px rgba(0,0,0,0.3); margin-bottom:12px;">
    <h1 style="margin:0; text-align:center;"> AI Customer Support Analyzer</h1>
</div>
"""

# After upload: icon beside title (inline)
_ICON_HTML_INLINE = f"""
<div style="display:flex; align-items:center; gap:14px; margin-bottom:12px;">
    <img src="data:image/jpeg;base64,{_b64}"
         style="width:64px; height:64px; border-radius:50%; object-fit:cover;
                box-shadow:0 2px 8px rgba(0,0,0,0.25); flex-shrink:0;">
    <h1 style="margin:0;"> AI Customer Support Analyzer</h1>
</div>
"""

# Use a placeholder so title always renders above the uploader
_title_slot = st.empty()
uploaded_file = st.file_uploader("Upload Support Tickets CSV")

if uploaded_file:
    _title_slot.markdown(_ICON_HTML_INLINE, unsafe_allow_html=True)
else:
    _title_slot.markdown(_ICON_HTML_CENTER, unsafe_allow_html=True)

if uploaded_file:

    # Load data
    df = load_data(uploaded_file)
    df = add_features(df)
    df = process_predictions(df)
    df["created_at"] = pd.to_datetime(df["created_at"])

    # Assign priority if not already in CSV
    if "priority" not in df.columns:
        def _priority(row):
            if row["urgent"] and row["sentiment"] == "Negative":
                return "high"
            elif row["urgent"] or row["sentiment"] == "Negative" or row["category"] in ("Billing", "Technical"):
                return "medium"
            return "low"
        df["priority"] = df.apply(_priority, axis=1)

    _msg = st.empty()
    _msg.success("Data processed successfully!")
    time.sleep(5)
    _msg.empty()

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

    priority_filter = st.sidebar.selectbox(
        "Priority",
        ["All", "high", "medium", "low"]
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

    if priority_filter != "All":
        filtered_df = filtered_df[filtered_df["priority"] == priority_filter]

    st.sidebar.markdown(f"**Showing {len(filtered_df)} of {len(df)} tickets**")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚨 Alerts**")
    alerts = generate_alerts(filtered_df)

    if alerts:
        for level, message in alerts:
            if level == "error":
                st.sidebar.error(message)
            elif level == "warning":
                st.sidebar.warning(message)
            else:
                st.sidebar.info(message)
    else:
        st.success("✅ No major alert conditions detected in the filtered ticket set.")

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Results",
        data=csv_data,
        file_name="filtered_support_tickets.csv",
        mime="text/csv"
    )
    if st.sidebar.button("Send Alert for High Priority Tickets"):
        try:
            sent = send_high_priority_alert(filtered_df)
            if sent:
                st.sidebar.success(f"Alert sent for {sent} high priority ticket(s)!")
            else:
                st.sidebar.info("No high priority tickets to alert on.")
        except Exception as e:
            st.sidebar.error(f"Failed to send alert: {e}")

    # ---------------------------
    # TABS
    # ---------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "📋 Data",
        "🤖 AI Insights",
        "📈 Forecast",
        "🔵 Clusters"
    ])

    # ---------------------------
    # TAB 1: DASHBOARD
    # ---------------------------
    with tab1:
        st.subheader("Ticket Overview")        

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Tickets", len(filtered_df))
        col2.metric("Urgent Tickets", filtered_df["urgent"].sum())
        col3.metric("Negative Sentiment", (filtered_df["sentiment"] == "Negative").sum())
        col4.metric("High Priority Tickets", (filtered_df["priority"] == "high").sum())

        # high_count = (filtered_df["priority"] == "high").sum()
        # st.metric("High Priority Tickets", high_count)

        ch_col1, ch_col2 = st.columns(2)

        with ch_col1:
            st.subheader("Category Distribution")
            st.bar_chart(filtered_df["category"].value_counts())

        with ch_col2:
            st.subheader("Sentiment Distribution")
            fig_pie = px.pie(
                values=filtered_df["sentiment"].value_counts().values,
                names=filtered_df["sentiment"].value_counts().index,
                hole=0.3
            )
            st.plotly_chart(fig_pie, use_container_width=True)

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

    # ---------------------------
    # TAB 4: FORECAST
    # ---------------------------
    with tab4:
        st.subheader("🔮 Ticket Volume Forecast")

        col_a, col_b = st.columns(2)
        forecast_category = col_a.selectbox(
            "Filter by Category",
            ["All"] + sorted(df["category"].unique().tolist()),
            key="forecast_cat"
        )
        days_ahead = col_b.slider("Days to Forecast", min_value=1, max_value=30, value=7)

        history, forecast_df = forecast_tickets(df, days_ahead=days_ahead, category=forecast_category)

        if forecast_df is None:
            st.warning("Not enough historical data to generate a forecast (need at least 3 days).")
        else:
            # Combined chart: history + forecast
            history_plot = history.set_index("date")["count"].rename("Historical")
            forecast_plot = forecast_df.set_index("date")["predicted_tickets"].rename("Forecast")
            combined = pd.concat([history_plot, forecast_plot], axis=1)
            st.line_chart(combined)

            # Tomorrow highlight
            tomorrow_val = int(forecast_df.iloc[0]["predicted_tickets"])
            st.metric(
                label=f"Predicted tickets tomorrow ({forecast_df.iloc[0]['date'].date()})",
                value=tomorrow_val
            )

            st.dataframe(
                forecast_df.rename(columns={"date": "Date", "predicted_tickets": "Predicted Tickets"})
                .assign(Date=lambda x: x["Date"].dt.date)
                .reset_index(drop=True)
            )

    # ---------------------------
    # TAB 5: CLUSTERING
    # ---------------------------
    with tab5:
        st.subheader("🔵 Ticket Clustering")
        st.caption("Groups tickets by category, sentiment, priority, urgency, response time, and text length.")

        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=4, value=3, key="n_clusters")
        color_by = st.selectbox("Color points by", ["cluster", "category", "sentiment", "priority"], key="color_by")

        cluster_df = run_clustering(filtered_df, n_clusters=n_clusters)

        fig = px.scatter(
            cluster_df,
            x="pca_x", y="pca_y",
            color=color_by,
            hover_data=["text", "category", "sentiment", "priority", "urgent", "response_time_hours"],
            labels={"pca_x": "PCA Component 1", "pca_y": "PCA Component 2"},
            title=f"Ticket Clusters (k={n_clusters}) — colored by {color_by}",
            height=520
        )
        fig.update_traces(marker=dict(size=6, opacity=0.75))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Cluster Summary**")
        summary = (
            cluster_df.groupby("cluster")
            .agg(
                Count=("text", "count"),
                Top_Category=("category", lambda x: x.value_counts().index[0]),
                Top_Sentiment=("sentiment", lambda x: x.value_counts().index[0]),
                Top_Priority=("priority", lambda x: x.value_counts().index[0]),
                Avg_Response=("response_time_hours", "mean")
            )
            .round(1)
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True)