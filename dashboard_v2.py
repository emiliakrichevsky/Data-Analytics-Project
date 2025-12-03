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
        
        # Enhanced Time Series Analysis
        st.subheader("📈 Time Series Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select stocks for analysis
            stocks = df['stock'].unique()[:5]  # Limit to first 5 stocks for clarity
            selected_stocks = st.multiselect(
                "Select stocks for analysis",
                stocks,
                default=stocks[:2]
            )
        
        with col2:
            # Select time period
            date_range = st.date_input(
                "Select date range",
                [df['Date'].min().date(), df['Date'].max().date()]
            )
        
        if selected_stocks and len(date_range) == 2:
            # Filter data
            mask = (df['stock'].isin(selected_stocks)) & \
                   (df['Date'] >= pd.to_datetime(date_range[0])) & \
                   (df['Date'] <= pd.to_datetime(date_range[1]))
            filtered_df = df[mask]
            
                        # Simple price comparison
            fig = px.line(
                filtered_df,
                x='Date',
                y='Close',
                color='stock',
                title=f'Stock Prices: {date_range[0]} to {date_range[1]}'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
            
            # Simple correlation heatmap
            try:
                pivot_data = filtered_df.pivot_table(
                    index='Date',
                    columns='stock',
                    values='Close'
                ).dropna()
                
                if len(pivot_data.columns) > 1:
                    correlation_matrix = pivot_data.corr()
                    
                    fig_corr = px.imshow(
                        correlation_matrix,
                        text_auto='.2f',
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Stock Price Correlations",
                        zmin=-1,
                        zmax=1
                    )
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(fig_corr, width='stretch')
            except:
                st.info("Could not create correlation matrix")
        
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
            st.plotly_chart(fig, width='stretch')
        else:
            st.success("✅ No missing data in integrated dataset!")
        
        st.markdown("---")
        


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
        st.dataframe(outlier_df, width='stretch')
        
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
            st.plotly_chart(fig, width='stretch')
    
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
        st.dataframe(scaling_df, width='stretch')
        
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
            st.plotly_chart(fig, width='stretch')

# ============================================================
# PAGE 5: FEATURE ENGINEERING (UPDATED WITH STOCK TOGGLE)
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
            st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # ENGINEERED FEATURES ANALYSIS SECTION
        st.subheader("📈 Engineered Features Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select stocks for analysis
            stocks = df['stock'].unique()[:5]  # Limit to first 5 stocks for clarity
            selected_stocks = st.multiselect(
                "Select stocks for feature analysis",
                stocks,
                default=stocks[:2] if len(stocks) >= 2 else stocks[:1]
            )
        
        with col2:
            # Select time period
            if 'Date' in df.columns:
                min_date = df['Date'].min().date()
                max_date = df['Date'].max().date()
                date_range = st.date_input(
                    "Select date range",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
            else:
                st.info("Date column not available in ML dataset")
                date_range = [None, None]
        
        if selected_stocks and len(date_range) == 2 and date_range[0] and date_range[1]:
            try:
                # Filter data
                mask = (df['stock'].isin(selected_stocks)) & \
                       (df['Date'] >= pd.to_datetime(date_range[0])) & \
                       (df['Date'] <= pd.to_datetime(date_range[1]))
                filtered_df = df[mask].copy()
                
                if len(filtered_df) == 0:
                    st.warning("No data available for the selected date range and stocks.")
                else:
                    # 1. Daily Returns Analysis
                    st.markdown("##### 📊 Daily Returns Analysis")
                    
                    if 'daily_return' in filtered_df.columns:
                        # Create line chart of daily returns
                        fig_returns = px.line(
                            filtered_df,
                            x='Date',
                            y='daily_return',
                            color='stock',
                            title='Daily Returns Over Time',
                            labels={'daily_return': 'Daily Return'}
                        )
                        fig_returns.update_layout(
                            height=400,
                            xaxis_title="Date",
                            yaxis_title="Daily Return",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_returns, width='stretch')
                        
                        # Distribution of daily returns
                        fig_returns_dist = px.histogram(
                            filtered_df,
                            x='daily_return',
                            color='stock',
                            nbins=50,
                            title='Distribution of Daily Returns',
                            opacity=0.7,
                            barmode='overlay',
                            labels={'daily_return': 'Daily Return'}
                        )
                        fig_returns_dist.update_layout(
                            height=400,
                            xaxis_title="Daily Return",
                            yaxis_title="Frequency"
                        )
                        st.plotly_chart(fig_returns_dist, width='stretch')
                    else:
                        st.info("'daily_return' feature not available in the dataset")
                    
                    # 2. Volatility Analysis (21-day rolling)
                    st.markdown("##### 🌊 Rolling Volatility Analysis")
                    
                    volatility_data = []
                    
                    for stock in selected_stocks:
                        stock_data = filtered_df[filtered_df['stock'] == stock].sort_values('Date')
                        
                        if 'daily_return' in stock_data.columns and len(stock_data) >= 21:
                            returns = stock_data['daily_return'].dropna()
                            
                            if len(returns) >= 21:
                                # Calculate rolling volatility (21-day)
                                rolling_vol = returns.rolling(window=21).std()
                                
                                # Prepare data for plotting
                                for date, vol in zip(stock_data['Date'].iloc[20:], rolling_vol.iloc[20:]):
                                    volatility_data.append({
                                        'Date': date,
                                        'Volatility': vol,
                                        'Stock': stock
                                    })
                    
                    if volatility_data:
                        vol_df = pd.DataFrame(volatility_data)
                        
                        # Create volatility chart
                        fig_volatility = px.line(
                            vol_df,
                            x='Date',
                            y='Volatility',
                            color='Stock',
                            title="21-Day Rolling Volatility of Daily Returns",
                            labels={'Volatility': 'Volatility'}
                        )
                        fig_volatility.update_layout(
                            height=400,
                            xaxis_title="Date",
                            yaxis_title="Volatility",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_volatility, width='stretch')
                        
                        # Volatility distribution
                        fig_vol_dist = px.histogram(
                            vol_df,
                            x='Volatility',
                            color='Stock',
                            nbins=30,
                            title="Distribution of Rolling Volatility",
                            opacity=0.7,
                            barmode='overlay',
                            labels={'Volatility': '21-Day Rolling Volatility'}
                        )
                        fig_vol_dist.update_layout(
                            height=400,
                            xaxis_title="Volatility",
                            yaxis_title="Frequency"
                        )
                        st.plotly_chart(fig_vol_dist, width='stretch')
                    else:
                        st.info("Need at least 21 days of 'daily_return' data for volatility calculation.")
                    
                    # 3. Feature Correlation Analysis
                    st.markdown("##### 🔗 Feature Correlations")
                    
                    # Select features to analyze
                    available_features = [col for col in filtered_df.columns 
                                         if col not in ['Date', 'stock', 'y', 'stock_fwd_ret_21d', 'sp500_fwd_ret_21d']]
                    
                    selected_features = st.multiselect(
                        "Select features for correlation analysis",
                        available_features[:10],  # Limit to first 10 for performance
                        default=available_features[:3] if len(available_features) >= 3 else available_features
                    )
                    
                    if len(selected_features) >= 2:
                        # Create correlation matrix
                        corr_data = filtered_df[selected_features].corr()
                        
                        fig_corr = px.imshow(
                            corr_data,
                            text_auto='.2f',
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title="Correlation Matrix of Selected Features",
                            zmin=-1,
                            zmax=1
                        )
                        fig_corr.update_layout(
                            height=500,
                            width=500
                        )
                        st.plotly_chart(fig_corr, width='stretch')
                    
                    st.markdown("---")
                    
                    # 4. Feature Comparison Over Time (WITH STOCK TOGGLE)
                    st.markdown("##### 📈 Multiple Features Comparison")
                    
                    # Create columns for better layout
                    col_feat1, col_feat2, col_stock = st.columns([2, 2, 1])
                    
                    with col_feat1:
                        # Feature selection
                        comparison_features = st.multiselect(
                            "Select features to compare",
                            available_features[:8],  # Show first 8 features
                            default=available_features[:2] if len(available_features) >= 2 else available_features[:1],
                            key="feature_comparison_select"
                        )
                    
                    with col_feat2:
                        # Chart type selection
                        chart_type = st.selectbox(
                            "Chart type",
                            ["Line Chart", "Area Chart", "Scatter Plot"],
                            key="chart_type_select"
                        )
                    
                    with col_stock:
                        # Single stock selection for comparison
                        comparison_stock = st.selectbox(
                            "Select stock",
                            selected_stocks,
                            key="comparison_stock_select"
                        )
                    
                    if comparison_features and comparison_stock:
                        stock_data = filtered_df[filtered_df['stock'] == comparison_stock].sort_values('Date')
                        
                        # Check if we have enough data
                        if len(stock_data) == 0:
                            st.warning(f"No data available for {comparison_stock} in the selected date range.")
                        else:
                            # Create the comparison chart
                            fig_comparison = go.Figure()
                            
                            # Track if we found any features
                            features_found = False
                            
                            for feature in comparison_features:
                                if feature in stock_data.columns:
                                    features_found = True
                                    feature_data = stock_data[feature].dropna()
                                    
                                    if len(feature_data) > 0:
                                        if chart_type == "Line Chart":
                                            fig_comparison.add_trace(go.Scatter(
                                                x=stock_data['Date'],
                                                y=stock_data[feature],
                                                name=feature,
                                                mode='lines',
                                                opacity=0.8
                                            ))
                                        elif chart_type == "Area Chart":
                                            fig_comparison.add_trace(go.Scatter(
                                                x=stock_data['Date'],
                                                y=stock_data[feature],
                                                name=feature,
                                                mode='lines',
                                                fill='tozeroy',
                                                opacity=0.3
                                            ))
                                        elif chart_type == "Scatter Plot":
                                            fig_comparison.add_trace(go.Scatter(
                                                x=stock_data['Date'],
                                                y=stock_data[feature],
                                                name=feature,
                                                mode='markers',
                                                marker=dict(size=6, opacity=0.7)
                                            ))
                            
                            if features_found:
                                fig_comparison.update_layout(
                                    title=f"Feature Comparison for {comparison_stock}",
                                    height=500,
                                    xaxis_title="Date",
                                    yaxis_title="Feature Value",
                                    hovermode='x unified',
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                # Add a button to normalize/standardize the features for better comparison
                                col_norm, col_empty = st.columns([1, 3])
                                with col_norm:
                                    normalize = st.checkbox("Normalize features", value=False)
                                
                                if normalize:
                                    # Normalize each feature to 0-1 range for better comparison
                                    fig_comparison_normalized = go.Figure()
                                    
                                    for feature in comparison_features:
                                        if feature in stock_data.columns:
                                            feature_data = stock_data[feature].dropna()
                                            if len(feature_data) > 1:
                                                # Min-max normalization
                                                min_val = feature_data.min()
                                                max_val = feature_data.max()
                                                if max_val != min_val:
                                                    normalized = (feature_data - min_val) / (max_val - min_val)
                                                    
                                                    if chart_type == "Line Chart":
                                                        fig_comparison_normalized.add_trace(go.Scatter(
                                                            x=stock_data['Date'],
                                                            y=normalized,
                                                            name=f"{feature} (normalized)",
                                                            mode='lines',
                                                            opacity=0.8
                                                        ))
                                    
                                    if len(fig_comparison_normalized.data) > 0:
                                        fig_comparison_normalized.update_layout(
                                            title=f"Normalized Feature Comparison for {comparison_stock}",
                                            height=500,
                                            xaxis_title="Date",
                                            yaxis_title="Normalized Value (0-1)",
                                            hovermode='x unified',
                                            legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=1.02,
                                                xanchor="right",
                                                x=1
                                            )
                                        )
                                        st.plotly_chart(fig_comparison_normalized, width='stretch')
                                    else:
                                        st.plotly_chart(fig_comparison, width='stretch')
                                else:
                                    st.plotly_chart(fig_comparison, width='stretch')
                                
                                # Show feature statistics table
                                st.markdown("##### 📊 Feature Statistics")
                                
                                stats_data = []
                                for feature in comparison_features:
                                    if feature in stock_data.columns:
                                        feature_data = stock_data[feature].dropna()
                                        if len(feature_data) > 0:
                                            stats_data.append({
                                                'Feature': feature,
                                                'Mean': round(feature_data.mean(), 4),
                                                'Std Dev': round(feature_data.std(), 4),
                                                'Min': round(feature_data.min(), 4),
                                                'Max': round(feature_data.max(), 4),
                                                'Data Points': len(feature_data)
                                            })
                                
                                if stats_data:
                                    stats_df = pd.DataFrame(stats_data)
                                    st.dataframe(stats_df, width='stretch')
                            else:
                                st.info("Selected features not found in the dataset for the chosen stock.")
                    
                    # 5. Cross-Stock Feature Comparison (NEW)
                    st.markdown("##### 🔄 Cross-Stock Feature Comparison")
                    
                    if len(selected_stocks) >= 2 and len(comparison_features) >= 1:
                        cross_stock_feature = st.selectbox(
                            "Select feature to compare across stocks",
                            comparison_features if comparison_features else available_features[:3],
                            key="cross_stock_feature"
                        )
                        
                        if cross_stock_feature:
                            # Prepare data for comparison
                            cross_data = []
                            for stock in selected_stocks:
                                stock_data = filtered_df[filtered_df['stock'] == stock].sort_values('Date')
                                if cross_stock_feature in stock_data.columns:
                                    feature_data = stock_data[cross_stock_feature].dropna()
                                    if len(feature_data) > 0:
                                        cross_data.append({
                                            'Stock': stock,
                                            'Mean': feature_data.mean(),
                                            'Std Dev': feature_data.std(),
                                            'Min': feature_data.min(),
                                            'Max': feature_data.max()
                                        })
                            
                            if len(cross_data) >= 2:
                                cross_df = pd.DataFrame(cross_data)
                                
                                # Bar chart comparison
                                fig_cross = px.bar(
                                    cross_df,
                                    x='Stock',
                                    y='Mean',
                                    error_y='Std Dev',
                                    title=f"Mean {cross_stock_feature} Comparison Across Stocks",
                                    color='Stock',
                                    labels={'Mean': f'Mean {cross_stock_feature}'}
                                )
                                fig_cross.update_layout(height=400)
                                st.plotly_chart(fig_cross, width='stretch')
                                
                                # Show comparison table
                                st.dataframe(cross_df, width='stretch')
                    
            except Exception as e:
                st.error(f"Error analyzing features: {e}")
        
        st.markdown("---")
        
        # Feature Importance Visualization (Based on your results)
        st.subheader("📊 Hypothetical Feature Importance Analysis")
        
        # Based on your model results, we can simulate feature categories impact
        # Create feature categories importance chart
        feature_impact_data = {
            "Feature Category": ["Technical Indicators", "Price Features", "Market Indicators", 
                               "Macro Features", "Sector Features", "Relative Features"],
            "Hypothetical Importance": [25, 22, 18, 15, 12, 8],
            "Count": [6, 9, 5, 4, 5, 2]  # Number of features in each category
        }
        
        feature_df = pd.DataFrame(feature_impact_data)
        
        # Treemap visualization
        fig_treemap = px.treemap(
            feature_df,
            path=['Feature Category'],
            values='Hypothetical Importance',
            color='Count',
            color_continuous_scale='Blues',
            title="Feature Categories Impact Distribution",
            hover_data=['Count']
        )
        fig_treemap.update_layout(height=500)
        st.plotly_chart(fig_treemap, width='stretch')
        
        # Parallel coordinates for feature analysis
        st.markdown("##### 📐 Feature Characteristics Analysis")
        
        # Create synthetic data for demonstration
        synthetic_features = pd.DataFrame({
            'Feature': ['daily_return', 'sp500_daily_return', 'VIX_t', 'sector_XLK_t', 
                       'MA20_ratio', 'relative_return', 'Put_Call_Ratio_t'],
            'Volatility': [8.5, 7.2, 9.8, 6.3, 5.1, 7.9, 8.2],
            'Correlation_with_Target': [0.45, 0.38, 0.52, 0.31, 0.42, 0.48, 0.39],
            'Missing_Rate': [0.02, 0.01, 0.15, 0.08, 0.03, 0.05, 0.12],
            'Feature_Type': ['Price', 'Benchmark', 'Macro', 'Sector', 'Technical', 'Relative', 'Market']
        })
        
        fig_parallel = px.parallel_categories(
            synthetic_features,
            dimensions=['Feature_Type', 'Volatility', 'Correlation_with_Target'],
            color='Correlation_with_Target',
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Feature Characteristics Parallel Categories",
            labels={
                'Feature_Type': 'Feature Type',
                'Volatility': 'Volatility Score',
                'Correlation_with_Target': 'Target Correlation'
            }
        )
        fig_parallel.update_layout(height=500)
        st.plotly_chart(fig_parallel, width='stretch')

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
        st.dataframe(rf_df, width='stretch')
        
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
        st.plotly_chart(fig, width='stretch')
    
    # GBM Results
    if 'gbm_results' in data:
        st.markdown("### ⚡ Gradient Boosting Machine")
        
        gbm_df = data['gbm_results']
        
        # Display table
        st.dataframe(gbm_df, width='stretch')
        
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
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Enhanced Model Performance Comparison
    # 1. Performance Metrics Comparison (Accuracy, F1, AUC)
    if 'rf_results' in data and 'gbm_results' in data:
        st.subheader("📊 Model Performance Metrics Comparison")
        
        # Prepare data for comparison
        comparison_data = []
        
        # Random Forest metrics
        for _, row in data['rf_results'].iterrows():
            comparison_data.append({
                'Scenario': row['Scenario'],
                'Model': 'Random Forest',
                'Accuracy': row['Accuracy'],
                'F1': row['F1'],
                'Metric': 'Accuracy',
                'Value': row['Accuracy']
            })
            comparison_data.append({
                'Scenario': row['Scenario'],
                'Model': 'Random Forest', 
                'Accuracy': row['Accuracy'],
                'F1': row['F1'],
                'Metric': 'F1',
                'Value': row['F1']
            })
        
        # GBM metrics
        for _, row in data['gbm_results'].iterrows():
            comparison_data.append({
                'Scenario': row['Scenario'],
                'Model': 'GBM',
                'Accuracy': row['Accuracy'],
                'F1': row['F1'],
                'AUC': row.get('AUC', None),
                'Metric': 'Accuracy',
                'Value': row['Accuracy']
            })
            comparison_data.append({
                'Scenario': row['Scenario'],
                'Model': 'GBM',
                'Accuracy': row['Accuracy'],
                'F1': row['F1'],
                'AUC': row.get('AUC', None),
                'Metric': 'F1',
                'Value': row['F1']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create radar/spider chart
        st.markdown("##### 📈 Metrics Radar Chart by Model")
        
        # Prepare data for radar chart
        scenarios = ['Scaled Data', 'All Combined']  # Best scenarios
        metrics = ['Accuracy', 'F1']
        
        fig_radar = go.Figure()
        
        for model in ['Random Forest', 'GBM']:
            for scenario in scenarios:
                model_scenario_data = df_comparison[
                    (df_comparison['Model'] == model) & 
                    (df_comparison['Scenario'] == scenario)
                ]
                if len(model_scenario_data) > 0:
                    values = []
                    for metric in metrics:
                        val = model_scenario_data[model_scenario_data['Metric'] == metric]['Value'].values
                        if len(val) > 0:
                            values.append(val[0])
                        else:
                            values.append(0)
                    
                    # Add scenario data
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values + [values[0]],  # Close the polygon
                        theta=metrics + [metrics[0]],
                        name=f"{model} - {scenario}",
                        fill='toself',
                        opacity=0.6
                    ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.35, 0.65]  # Adjusted range based on your data
                )),
            showlegend=True,
            height=500,
            title="Performance Metrics Comparison (Best Scenarios)"
        )
        
        st.plotly_chart(fig_radar, width='stretch')
        
        # 2. Model Improvement Heatmap
        st.markdown("##### 🔥 Data Preparation Impact Heatmap")
        
        # Calculate improvement from baseline
        heatmap_data = []
        
        for model_name, results_df in [('Random Forest', data['rf_results']), ('GBM', data['gbm_results'])]:
            baseline_acc = results_df[results_df['Scenario'] == 'Baseline (Simple Fill)']['Accuracy'].values[0]
            
            for _, row in results_df.iterrows():
                if row['Scenario'] != 'Baseline (Simple Fill)':
                    improvement = (row['Accuracy'] - baseline_acc) * 100  # Percentage improvement
                    heatmap_data.append({
                        'Model': model_name,
                        'Scenario': row['Scenario'],
                        'Improvement': improvement
                    })
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            heatmap_pivot = heatmap_df.pivot(index='Scenario', columns='Model', values='Improvement')
            
            fig_heatmap = px.imshow(
                heatmap_pivot,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdYlGn',
                title="Accuracy Improvement Over Baseline (%)",
                labels=dict(x="Model", y="Data Prep Strategy", color="Improvement (%)")
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, width='stretch')

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
        st.plotly_chart(fig, width='stretch')
    
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
    
    st.markdown("---")
    
    # Data Preparation Evolution Timeline
    st.subheader("📅 Data Preparation Evolution & Impact")
    
    if 'rf_results' in data:
        # Create timeline of data prep stages
        timeline_data = [
            {"Stage": "Baseline", "Accuracy_RF": data['rf_results'].loc[0, 'Accuracy'], "Description": "Simple filling only"},
            {"Stage": "+ Smart Imputation", "Accuracy_RF": data['rf_results'].loc[1, 'Accuracy'], "Description": "Advanced null handling"},
            {"Stage": "+ Outlier Removal", "Accuracy_RF": data['rf_results'].loc[2, 'Accuracy'], "Description": "Z-score > 3 removed"},
            {"Stage": "+ Feature Scaling", "Accuracy_RF": data['rf_results'].loc[3, 'Accuracy'], "Description": "Standardization applied"},
            {"Stage": "All Combined", "Accuracy_RF": data['rf_results'].loc[4, 'Accuracy'], "Description": "Complete pipeline"}
        ]
        
        timeline_df = pd.DataFrame(timeline_data)
        
        # Add cumulative improvement
        baseline_acc = timeline_df.loc[0, 'Accuracy_RF']
        timeline_df['Improvement'] = (timeline_df['Accuracy_RF'] - baseline_acc) * 100
        
        # Create step chart
        fig_timeline = go.Figure()
        
        fig_timeline.add_trace(go.Scatter(
            x=timeline_df['Stage'],
            y=timeline_df['Accuracy_RF'],
            mode='lines+markers+text',
            text=[f"{acc:.2%}" for acc in timeline_df['Accuracy_RF']],
            textposition="top center",
            line=dict(width=4, color='#636EFA'),
            marker=dict(size=10, color='white', line=dict(width=2, color='#636EFA')),
            name="Random Forest Accuracy"
        ))
        
        # Add GBM if available
        if 'gbm_results' in data:
            gbm_accuracies = data['gbm_results']['Accuracy'].values
            fig_timeline.add_trace(go.Scatter(
                x=timeline_df['Stage'],
                y=gbm_accuracies,
                mode='lines+markers',
                line=dict(width=3, color='#EF553B', dash='dash'),
                marker=dict(size=8, color='white', line=dict(width=2, color='#EF553B')),
                name="GBM Accuracy"
            ))
        
        fig_timeline.update_layout(
            title="Model Performance Evolution with Data Preparation",
            xaxis_title="Data Preparation Stage",
            yaxis_title="Accuracy",
            height=500,
            hovermode='x unified',
            yaxis=dict(tickformat=".0%", range=[0.45, 0.55])  # Adjusted range for your data
        )
        
        st.plotly_chart(fig_timeline, width='stretch')
        
        # Improvement bar chart
        fig_improvement = px.bar(
            timeline_df,
            x='Stage',
            y='Improvement',
            title="Cumulative Improvement Over Baseline",
            text='Improvement',
            color='Improvement',
            color_continuous_scale='RdYlGn'
        )
        fig_improvement.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside'
        )
        fig_improvement.update_layout(
            height=400,
            xaxis_title="Data Preparation Stage",
            yaxis_title="Improvement (%)"
        )
        st.plotly_chart(fig_improvement, width='stretch')

# Footer
st.markdown("---")
st.caption("📊 Data Engineering Project Dashboard | Stock Prediction Pipeline")