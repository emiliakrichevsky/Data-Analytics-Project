"""
Data Engineering Project Dashboard
Interactive visualization of data pipeline, cleaning, and ML results
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page config
st.set_page_config(
    page_title="Stock Prediction Data Pipeline",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Stock Prediction Data Engineering Pipeline")
st.markdown("**Predicting Stock Outperformance vs S&P 500**")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📋 Overview",
    "🔗 Data Integration", 
    "🧹 Data Cleaning",
    "⚖️ Feature Scaling",
    "🔧 Feature Engineering",
    "🤖 Model Results",
    "📈 Performance Analysis"
])

# Load data function
@st.cache_data
def load_data():
    """Load all datasets"""
    data = {}
    
    # Raw data
    if os.path.exists('data/integrated_raw_data.csv'):
        data['raw'] = pd.read_csv('data/integrated_raw_data.csv')
        data['raw']['Date'] = pd.to_datetime(data['raw']['Date'])
    
    # Prepared data
    if os.path.exists('data/integrated_prepared_data.csv'):
        data['prepared'] = pd.read_csv('data/integrated_prepared_data.csv')
        data['prepared']['Date'] = pd.to_datetime(data['prepared']['Date'])
    
    # ML dataset
    if os.path.exists('data/ml_features_and_labels_clean.csv'):
        data['ml'] = pd.read_csv('data/ml_features_and_labels_clean.csv')
        data['ml']['Date'] = pd.to_datetime(data['ml']['Date'])
    
    # Model results
    if os.path.exists('data/model_comparison_timeseries_results.csv'):
        data['rf_results'] = pd.read_csv('data/model_comparison_timeseries_results.csv')
    
    if os.path.exists('data/model_comparison_timeseries_gbm_results.csv'):
        data['gbm_results'] = pd.read_csv('data/model_comparison_timeseries_gbm_results.csv')
    
    # Outlier analysis
    if os.path.exists('data/outlier_detection_summary.csv'):
        data['outliers'] = pd.read_csv('data/outlier_detection_summary.csv')
    
    # Scaling log
    if os.path.exists('data/scaling_log.csv'):
        data['scaling'] = pd.read_csv('data/scaling_log.csv')
    
    return data

# Load all data
data = load_data()

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "📋 Overview":
    st.header("Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ML Task", "Binary Classification")
        st.info("🎯 Predict if stock outperforms S&P 500")
    
    with col2:
        if 'raw' in data:
            st.metric("Total Data Points", f"{len(data['raw']):,}")
            st.metric("Date Range", f"{data['raw']['Date'].min().date()} to {data['raw']['Date'].max().date()}")
    
    with col3:
        if 'ml' in data:
            st.metric("ML Features", len([c for c in data['ml'].columns if c not in ['Date', 'stock', 'y', 'stock_fwd_ret_21d', 'sp500_fwd_ret_21d']]))
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
    
    st.markdown("---")
    
    # Data sources
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

# ============================================================
# PAGE 2: DATA INTEGRATION
# ============================================================
elif page == "🔗 Data Integration":
    st.header("Data Integration & Schema")
    
    if 'raw' in data:
        df = data['raw']
        
        # Schema overview
        st.subheader("Integrated Schema")
        
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
            "S&P 500 Benchmark": [c for c in df.columns if c.startswith('sp500_')],
            "Macro Indicators": ["CPI", "GDP", "Fed_Funds_Rate", "Unemployment_Rate"],
            "Market Indicators": ["VIX", "Put_Call_Ratio", "Market_Breadth"],
            "Sector ETFs": [c for c in df.columns if c.startswith('sector_')],
            "Technical Indicators": [c for c in df.columns if 'RSI' in c or 'SMA' in c]
        }
        
        for category, cols in categories.items():
            matching_cols = [c for c in cols if c in df.columns]
            if matching_cols:
                with st.expander(f"**{category}** ({len(matching_cols)} columns)"):
                    st.write(", ".join(matching_cols))
        
        # Time series visualization
        st.subheader("Time Series Coverage")
        
        # Stock prices over time
        fig = px.line(
            df, 
            x='Date', 
            y='Close', 
            color='stock',
            title='Stock Prices Over Time'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data completeness heatmap
        st.subheader("Data Completeness by Source")
        
        null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        null_pct = null_pct[null_pct > 0]
        
        if len(null_pct) > 0:
            fig = px.bar(
                x=null_pct.index,
                y=null_pct.values,
                title="Missing Data by Column (%)",
                labels={'x': 'Column', 'y': 'Missing %'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing data in integrated dataset!")

# ============================================================
# PAGE 3: DATA CLEANING
# ============================================================
elif page == "🧹 Data Cleaning":
    st.header("Data Cleaning Process")
    
    # Null handling
    st.subheader("1️⃣ Null Value Handling")
    
    if 'raw' in data and 'prepared' in data:
        raw_nulls = data['raw'].isnull().sum().sum()
        prep_nulls = data['prepared'].isnull().sum().sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nulls Before", f"{raw_nulls:,}")
        with col2:
            st.metric("Nulls After", f"{prep_nulls:,}")
        with col3:
            improvement = ((raw_nulls - prep_nulls) / raw_nulls * 100) if raw_nulls > 0 else 100
            st.metric("Improvement", f"{improvement:.1f}%")
        
        st.info("**Strategy**: Smart imputation - Forward/backward fill for macro data, forward fill for stock prices, linear interpolation for scattered missing values")
    
    # Outlier detection
    st.subheader("2️⃣ Outlier Detection & Removal")
    
    if 'outliers' in data:
        outlier_df = data['outliers']
        
        # Display outlier summary
        st.dataframe(outlier_df, use_container_width=True)
        
        # Visualization
        if 'column' in outlier_df.columns and 'outliers_detected' in outlier_df.columns:
            top_outliers = outlier_df.nlargest(10, 'outliers_detected')
            
            fig = px.bar(
                top_outliers,
                x='column',
                y='outliers_detected',
                title='Top 10 Columns by Outliers Detected (Z-score > 3)',
                labels={'outliers_detected': 'Number of Outliers'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    if 'raw' in data and 'prepared' in data:
        rows_before = len(data['raw'])
        rows_after = len(data['prepared'])
        removed = rows_before - rows_after
        
        st.metric("Rows Removed", f"{removed:,} ({removed/rows_before*100:.2f}%)")
        st.info("**Method**: Z-score threshold = 3 standard deviations")

# ============================================================
# PAGE 4: FEATURE SCALING
# ============================================================
elif page == "⚖️ Feature Scaling":
    st.header("Feature Scaling & Normalization")
    
    if 'scaling' in data:
        scaling_df = data['scaling']
        
        st.subheader("Scaling Strategies")
        
        # Group by scaler
        if 'scaler' in scaling_df.columns:
            scaler_counts = scaling_df['scaler'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("StandardScaler", scaler_counts.get('StandardScaler', 0))
                st.caption("For normally distributed features")
            
            with col2:
                st.metric("RobustScaler", scaler_counts.get('RobustScaler', 0))
                st.caption("For features with outliers")
            
            with col3:
                st.metric("MinMaxScaler", scaler_counts.get('MinMaxScaler', 0))
                st.caption("For bounded features")
        
        # Display scaling log
        st.subheader("Scaling Details")
        st.dataframe(scaling_df, use_container_width=True)
        
        # Before/After comparison (if available)
        if 'mean_before' in scaling_df.columns and 'mean_after' in scaling_df.columns:
            st.subheader("Mean Normalization Effect")
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Before Scaling", "After Scaling"])
            
            # Before
            fig.add_trace(
                go.Bar(x=scaling_df['column'][:10], y=scaling_df['mean_before'][:10], name="Before"),
                row=1, col=1
            )
            
            # After
            fig.add_trace(
                go.Bar(x=scaling_df['column'][:10], y=scaling_df['mean_after'][:10], name="After"),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 5: FEATURE ENGINEERING
# ============================================================
elif page == "🔧 Feature Engineering":
    st.header("Feature Engineering")
    
    if 'ml' in data:
        df = data['ml']
        
        # Feature categories
        st.subheader("Engineered Features (30+ features)")
        
        feature_categories = {
            "Stock Price Features": ["daily_return", "r_1W", "r_1M", "r_3M", "vol_1M", "MA20_ratio", "MA50_ratio", "HL_range", "vol_z"],
            "S&P 500 Features": ["sp500_daily_return", "sp500_vol_1M"],
            "Relative Features": ["relative_return", "volatility_ratio"],
            "Macro Features": ["VIX_t", "FedFunds_t", "CPI_chg", "FedFunds_chg"],
            "Market Features": ["Put_Call_Ratio_t", "Market_Breadth_t"],
            "Technical Indicators": ["spy_RSI_t", "spy_SMA_50_t", "spy_SMA_200_t", "qqq_RSI_t", "qqq_SMA_50_t", "qqq_SMA_200_t"],
            "Sector Features": ["sector_XLK_t", "sector_XLF_t", "sector_XLV_t", "sector_XLE_t", "sector_XLI_t"]
        }
        
        for category, features in feature_categories.items():
            available = [f for f in features if f in df.columns]
            if available:
                with st.expander(f"**{category}** ({len(available)} features)"):
                    st.write(", ".join(available))
        
        # Target variable distribution
        st.subheader("Target Variable Distribution")
        
        if 'y' in df.columns:
            y_counts = df['y'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Outperform (y=1)", f"{y_counts.get(1, 0):,}")
            
            with col2:
                st.metric("Underperform (y=0)", f"{y_counts.get(0, 0):,}")
            
            with col3:
                balance = y_counts.get(1, 0) / (y_counts.get(0, 0) + y_counts.get(1, 0)) * 100
                st.metric("Balance", f"{balance:.1f}%")
            
            # Pie chart
            fig = px.pie(
                values=y_counts.values,
                names=['Underperform', 'Outperform'],
                title='Target Class Distribution',
                color_discrete_sequence=['#EF553B', '#00CC96']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 6: MODEL RESULTS
# ============================================================
elif page == "🤖 Model Results":
    st.header("Model Performance Comparison")
    
    st.subheader("Time-Series Split Results (Realistic)")
    
    # Random Forest Results
    if 'rf_results' in data:
        st.markdown("### 🌲 Random Forest")
        
        rf_df = data['rf_results']
        
        # Display table
        st.dataframe(rf_df, use_container_width=True)
        
        # Bar chart
        fig = px.bar(
            rf_df,
            x='Scenario',
            y='Accuracy',
            title='Random Forest: Data Prep Impact on Accuracy',
            text='Accuracy',
            color='Accuracy',
            color_continuous_scale='RdYlGn'
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # GBM Results
    if 'gbm_results' in data:
        st.markdown("### ⚡ Gradient Boosting Machine")
        
        gbm_df = data['gbm_results']
        
        # Display table
        st.dataframe(gbm_df, use_container_width=True)
        
        # Bar chart
        fig = px.bar(
            gbm_df,
            x='Scenario',
            y='AUC',
            title='GBM: Data Prep Impact on AUC',
            text='AUC',
            color='AUC',
            color_continuous_scale='RdYlGn'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 7: PERFORMANCE ANALYSIS
# ============================================================
elif page == "📈 Performance Analysis":
    st.header("Performance Analysis")
    
    st.subheader("Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ **Random Forest Best**: 51.8% (All Combined)")
        st.info("💡 Benefits from combining all data prep techniques")
    
    with col2:
        st.success("✅ **GBM Best**: 51.9% (Scaled Data Only)")
        st.info("💡 Performs best with scaling alone")
    
    st.markdown("---")
    
    # Comparison chart
    if 'rf_results' in data and 'gbm_results' in data:
        st.subheader("Model Comparison by Data Prep Scenario")
        
        # Merge results
        rf = data['rf_results'].copy()
        gbm = data['gbm_results'].copy()
        
        rf['Model'] = 'Random Forest'
        gbm['Model'] = 'GBM'
        
        # Align columns
        if 'AUC' not in rf.columns:
            rf['AUC'] = rf['Accuracy']  # Use accuracy as proxy
        
        combined = pd.concat([
            rf[['Scenario', 'Accuracy', 'Model']],
            gbm[['Scenario', 'Accuracy', 'Model']]
        ])
        
        fig = px.bar(
            combined,
            x='Scenario',
            y='Accuracy',
            color='Model',
            barmode='group',
            title='Random Forest vs GBM: Accuracy by Data Prep Strategy',
            color_discrete_map={'Random Forest': '#636EFA', 'GBM': '#EF553B'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Key insights
    st.subheader("📊 Data Preparation Impact")
    
    insights = [
        {"Technique": "Smart Imputation", "Impact": "Minimal (-0.6%)", "Insight": "Basic filling vs smart strategies"},
        {"Technique": "Outlier Removal", "Impact": "Slight Negative (-0.8%)", "Insight": "Removed valuable volatility signals"},
        {"Technique": "Feature Scaling", "Impact": "Positive (+1.6%)", "Insight": "Most effective single technique"},
        {"Technique": "All Combined", "Impact": "Best Overall (+3%)", "Insight": "Synergistic effect for Random Forest"}
    ]
    
    st.table(pd.DataFrame(insights))
    
    st.success("""
    **Key Takeaway**: Scaling is the most critical data preparation step for this time-series prediction task. 
    While Random Forest benefits from combining all techniques (51.8%), GBM achieves similar performance (51.9%) 
    with scaling alone, suggesting diminishing returns from additional preprocessing.
    """)

# Footer
st.markdown("---")
st.caption("📊 Data Engineering Project Dashboard | Stock Prediction Pipeline")
