import plotly
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.tools as tls
from utils import get_data, create_page_header, get_data_distribution, plot_data_distribution, create_box_plot

st.set_page_config(
    page_title="Data Distribution",
    page_icon="📊",
    layout="wide"
)

create_page_header("📊 Data Distribution Analysis")

data = get_data()

if not data or 'prepared' not in data:
    st.warning("No prepared data available. Please run data preparation first.")
else:
    df = data['prepared']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Dataset")
        dataset_option = st.radio(
            "Choose data to analyze:",
            ["Prepared Data", "ML Features"]
        )
    
    with col2:
        st.subheader("Select Column")
        if dataset_option == "Prepared Data":
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        else:
            numeric_cols = data['ml'].select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        selected_col = st.selectbox("Choose a column:", numeric_cols)
    
    st.markdown("---")
    
    # Distribution Statistics
    st.subheader("📈 Quick Statistics")
    
    if dataset_option == "Prepared Data":
        stats = get_data_distribution(df, selected_col)
        analyze_df = df
    else:
        stats = get_data_distribution(data['ml'], selected_col)
        analyze_df = data['ml']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{stats['mean']:.2f}")
    with col2:
        st.metric("Median", f"{stats['median']:.2f}")
    with col3:
        st.metric("Std Dev", f"{stats['std']:.2f}")
    with col4:
        st.metric("Range", f"{stats['min']:.2f} - {stats['max']:.2f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Q1 (25%)", f"{stats['q25']:.2f}")
    with col2:
        st.metric("Q3 (75%)", f"{stats['q75']:.2f}")

    st.write(analyze_df[selected_col].describe())
    st.markdown("---")
    
    # Distribution Plot
    st.subheader("📉 Distribution Histogram")
    try:
        fig = plot_data_distribution(analyze_df, selected_col, bins=40)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error plotting distribution: {e}")
    
    # Box Plot
    st.subheader("📦 Box Plot")
    try:
        # fig, ax = plt.subplots(figsize=(10, 4))
        # analyze_df[selected_col].dropna().plot(kind='box', ax=ax, vert=False)
        # ax.set_title(f"Box Plot of {selected_col}")
        # st.pyplot(fig)
        fig = create_box_plot(analyze_df, selected_col)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error plotting box plot: {e}")
    
    # # Data Summary
    # st.subheader("📋 Data Summary")
    # st.write(analyze_df[selected_col].describe())
