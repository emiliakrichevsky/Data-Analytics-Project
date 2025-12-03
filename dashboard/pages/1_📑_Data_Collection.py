import streamlit as st
import pandas as pd
from utils import load_data

st.set_page_config(
    page_title="Data Collection",
    page_icon="📑",
    layout="wide"
)
st.markdown("# Data Collection")
st.sidebar.header("Data Collection")
data = load_data()

st.subheader(":material/api: APIs")
st.markdown('''The datasets used in this project were collected from a combination of financial, macroeconomic, and technical-analysis data providers. Together, they capture price behavior, market conditions, and economic context necessary to predict whether **META** will outperform the **S&P 500**. All sources were accessed through APIs using the Python libraries `yfinance`, `requests`, and `pandas`, and all data are time-aligned by the variable `Date`''')

st.subheader("📁 Data Sources Used")
    
sources = [
    {"Category": "Stock Prices", "Files": "GOOGL_raw.csv, META_raw.csv", "Records": "5,000+"},
    {"Category": "Benchmark", "Files": "^GSPC_raw.csv (S&P 500)", "Records": "5,000+"},
    {"Category": "Macro Data", "Files": "CPI, GDP, Fed Funds Rate, Unemployment", "Records": "200+"},
    {"Category": "Market Indicators", "Files": "VIX, Put/Call Ratio, Market Breadth", "Records": "1,000+"},
    {"Category": "Sector ETFs", "Files": "XLK, XLF, XLV, XLE, XLI", "Records": "1,000+"},
    {"Category": "Technical Indicators", "Files": "SPY/QQQ RSI, SMA 50/200", "Records": "1,000+"}
]

st.table(pd.DataFrame(sources))