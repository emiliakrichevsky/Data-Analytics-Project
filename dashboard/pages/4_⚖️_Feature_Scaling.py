import streamlit as st
import time
import numpy as np
import pandas as pd
from utils import load_data
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Feature Scaling", page_icon="🔄", layout="wide")
st.markdown("# Feature Scaling")
st.sidebar.header("Feature Scaling")

data = load_data()

st.header("Feature Scaling & Normalization")
    
if 'scaling' in data:
    scaling_df = data['scaling']
    
    st.subheader("Techniques Applied")
    st.text("We applied different scalers based on the distribution of features:")
    # Group by scaler
    if 'method' in scaling_df.columns:
        scaler_counts = scaling_df['method'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("StandardScaler", scaler_counts.get('StandardScaler', 0))
            st.caption("For normally distributed features (e.g., Returns)")
        
        with col2:
            st.metric("RobustScaler", scaler_counts.get('RobustScaler', 0))
            st.caption("For features with outliers (e.g., Volume)")
        
        with col3:
            st.metric("MinMaxScaler", scaler_counts.get('MinMaxScaler', 0))
            st.caption("For bounded features (e.g., Ratios)")
    
    # Display scaling log
    st.subheader("Scaling Details")
    st.dataframe(scaling_df, width='stretch')
    
    # Before/After comparison (if available)
    if 'before_mean' in scaling_df.columns and 'after_mean' in scaling_df.columns:
        st.subheader("Mean Normalization Effect")
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Before Scaling", "After Scaling"])
        
        # Before
        fig.add_trace(
            go.Bar(x=scaling_df['column'][:10], y=scaling_df['before_mean'][:10], name="Before"),
            row=1, col=1
        )
        
        # After
        fig.add_trace(
            go.Bar(x=scaling_df['column'][:10], y=scaling_df['after_mean'][:10], name="After"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width='stretch')

