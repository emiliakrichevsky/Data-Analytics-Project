import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px
from utils import load_data

st.set_page_config(page_title="Data Integration", page_icon="🔗", layout="wide")
st.markdown("# Data Integration")
st.sidebar.header("Data Integration")

data = load_data()

st.header("Integrated Schema Overview")

if "raw" in data:
    df = data["raw"]

    # Schema overview
    # st.subheader("Integrated Schema")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Columns", len(df.columns))
        st.metric("Primary Key", "(stock, Date)")

    with col2:
        st.metric("Total Rows", f"{len(df):,}")
        st.metric("Null Values", f"{df.isnull().sum().sum():,}")

    # Column breakdown
    st.subheader("Column Categories")

    categories = {
        "Stock Data (OHLCV)": ["Close", "High", "Low", "Open", "Volume"],
        "S&P 500 Benchmark": [c for c in df.columns if c.startswith("sp500_")],
        "Macro Indicators": ["CPI", "GDP", "Fed_Funds_Rate", "Unemployment_Rate"],
        "Market Indicators": ["VIX", "Put_Call_Ratio", "Market_Breadth"],
        "Sector ETFs": [c for c in df.columns if c.startswith("sector_")],
        "Technical Indicators": [c for c in df.columns if "RSI" in c or "SMA" in c],
    }
    # Create a 2-column layout
    col1, col2 = st.columns(2)

    # Split categories between columns
    all_categories = list(categories.items())
    mid_point = len(all_categories) // 2

    for idx, (category, cols) in enumerate(all_categories):
        matching_cols = [c for c in cols if c in df.columns]
        if matching_cols:
            # Choose which column to use
            current_col = col1 if idx < mid_point else col2

            with current_col:
                # Create a card for each category
                st.markdown(
                    f"""
                <div style="
                    padding: 1.2rem;
                    margin-bottom: 1rem;
                    border-radius: 0.5rem;
                    background: #3F77C0;
                    color: white;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.8rem;">
                        <h4 style="margin: 0; font-size: 1.1rem;">{category}</h4>
                        <span style="
                            background-color: rgba(255,255,255,0.2);
                            padding: 0.2rem 0.6rem;
                            border-radius: 1rem;
                            font-size: 0.8rem;
                            font-weight: bold;
                        ">{len(matching_cols)} cols</span>
                    </div>
                    <div style="
                        padding: 0.8rem;
                        border-radius: 0.3rem;
                        font-size: 1.1rem;
                        line-height: 2;
                    ">
                        {', '.join(f'<code style="color:#FFFFFF; background-color: rgba(255,255,255,0.1); padding: 0.3rem 0.3rem; border-radius: 0.2rem; margin-right: 0.3rem;">{col}</code>' for col in matching_cols)}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.subheader("📈 Time Series Trends")

    col1, col2 = st.columns(2)

    with col1:
        # Select stocks for analysis
        stocks = df["stock"].unique()[:5]  # Limit to first 5 stocks for clarity
        selected_stocks = st.multiselect(
            "Select stocks for analysis", stocks, default=stocks[:2]
        )

    with col2:
        # Select time period
        date_range = st.date_input(
            "Select date range", [df["Date"].min().date(), df["Date"].max().date()]
        )

    if selected_stocks and len(date_range) == 2:
        # Filter data
        mask = (
            (df["stock"].isin(selected_stocks))
            & (df["Date"] >= pd.to_datetime(date_range[0]))
            & (df["Date"] <= pd.to_datetime(date_range[1]))
        )
        filtered_df = df[mask]

        # Simple price comparison
        fig = px.line(
            filtered_df,
            x="Date",
            y="Close",
            color="stock",
            title=f"Stock Prices: {date_range[0]} to {date_range[1]}",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
