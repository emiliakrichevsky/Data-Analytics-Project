import streamlit as st
import pandas as pd
from utils import load_data
st.set_page_config(
    page_title="Stock Prediction Data Pipeline",
    page_icon="📊",
)

st.write("# Project Overview")

st.sidebar.success("Select a pipeline stage above.")
data = load_data()
    
st.metric("ML Task", "Binary Classification")
st.info("🎯 Predict whether an individual stock will outperform S&P 500 index over the next 21 days (~ 1 month)")
if 'raw' in data:
    st.metric("Date Range", f"{data['raw']['Date'].min().date()} to {data['raw']['Date'].max().date()}")
col1, col2, col3 = st.columns(3)
with col1:
    if 'raw' in data:
        st.metric("Total Data Points", f"{len(data['raw']):,}")
with col2:
    if 'ml' in data:
        st.metric("ML Features", len([c for c in data['ml'].columns if c not in ['Date', 'stock', 'y', 'stock_fwd_ret_21d', 'sp500_fwd_ret_21d']]))
with col3:
    if 'ml' in data:
        st.metric("Stocks Analyzed", len(data['ml']['stock'].unique()))

st.markdown("---")

# Pipeline overview
st.subheader("Data Pipeline")

pipeline_steps = [
    "1️⃣ **Data Collection**: 10+ data sources (stocks, macro, market indicators)",
    "2️⃣ **Data Integration**: Merge on (stock, Date) key",
    "3️⃣ **Data Cleaning**: Smart imputation + outlier removal",
    "4️⃣ **Feature Scaling**: StandardScaler, RobustScaler, MinMaxScaler",
    "5️⃣ **Feature Engineering**: 30+ technical/fundamental features",
    "6️⃣ **Model Training**: Random Forest, GBM, H2O AutoML",
    "7️⃣ **Evaluation**: Time-series split (80/20)"
]

for step in pipeline_steps:
    st.markdown(step)
