import streamlit as st
import time
import numpy as np
import pandas as pd
from utils import load_data
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Feature Engineering", page_icon="🔧", layout="wide")
st.markdown("# Feature Engineering")
st.sidebar.header("Feature Engineering")
data = load_data()

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
    
    st.markdown("---")
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
    st.header("📈 Engineered Features Analysis")
    
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
                st.markdown("#### 1. Daily Returns Analysis")
                
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
                st.markdown("#### 2. Rolling Volatility Analysis")
                
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
                st.header("Feature Correlation and Comparison")
                
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
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlation Matrix of Selected Features",
                        zmin=-1,
                        zmax=1
                    )
                    fig_corr.update_traces(texttemplate='%{z:.2f}', textfont={'size':12})
                    fig_corr.update_layout(
                        height=500,
                        width=500
                    )
                    st.plotly_chart(fig_corr, width='stretch')
                
                st.markdown("---")
                
                # 4. Feature Comparison Over Time (WITH STOCK TOGGLE)
                st.subheader(" Multiple Features Comparison")
                
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
                                # n = fig_comparison_normalized.data
                                
                                if fig_comparison_normalized is not None and fig_comparison_normalized.data != ():
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
                            st.markdown("##### Feature Statistics")
                            
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
                
                # 5. Cross-Stock Feature Comparison 
                st.header("🔄 Cross-Stock Feature Comparison")
                
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
