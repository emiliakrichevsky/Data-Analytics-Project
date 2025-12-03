import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils import load_data

st.set_page_config(page_title="Performance Analysis", page_icon="📈", layout="wide")
st.markdown("# Performance Analysis")
st.sidebar.header("Performance Analysis")
data = load_data()
st.subheader("Key Findings")
    
col1, col2 = st.columns(2)

with col1:
    st.success("✅ **Random Forest Best**: 51.8% (All Combined)")
    st.info("💡 Benefits from combining all data prep techniques")

with col2:
    st.success("✅ **GBM Best**: 51.9% (Scaled Data Only)")
    st.info("💡 Performs best with scaling alone")

st.warning("""
**Key Takeaway**: Scaling is the most critical data preparation step for this time-series prediction task. 
While Random Forest benefits from combining all techniques (51.8%), GBM achieves similar performance (51.9%) 
with scaling alone, suggesting diminishing returns from additional preprocessing.
""")
st.markdown("---")

st.subheader("Model Comparison by Data Prep Scenario")
# Comparison chart
# if 'rf_results' in data and 'gbm_results' in data:
    
#     # Merge results
#     rf = data['rf_results'].copy()
#     gbm = data['gbm_results'].copy()
    
#     rf['Model'] = 'Random Forest'
#     gbm['Model'] = 'GBM'
    
#     # Align columns
#     if 'AUC' not in rf.columns:
#         rf['AUC'] = rf['Accuracy']  # Use accuracy as proxy
    
#     combined = pd.concat([
#         rf[['Scenario', 'Accuracy', 'Model']],
#         gbm[['Scenario', 'Accuracy', 'Model']]
#     ])
    
#     fig = px.bar(
#         combined,
#         x='Scenario',
#         y='Accuracy',
#         color='Model',
#         barmode='group',
#         title='Model Comparison: Accuracy by Scenario',
#         text='Accuracy',
#         color_discrete_map={'Random Forest': "#636EFA", 'GBM': '#EF553B'}
#     )
#     fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
#     fig.update_layout(height=500)
#     st.plotly_chart(fig, width='stretch')

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
    baseline_acc = timeline_df['Accuracy_RF'].iloc[0]
    timeline_df['Improvement'] = (timeline_df['Accuracy_RF'] - baseline_acc) * 100
    
    # Create step chart
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Scatter(
        x=timeline_df['Stage'],
        y=timeline_df['Accuracy_RF'],
        mode='lines+markers+text',
        text=[f"{acc:.2%}" for acc in timeline_df['Accuracy_RF']],
        textposition="top center",
        line=dict(width=4, color="#3F77C0"),
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
        title="Evolution of Model Performance with Data Preparation",
        xaxis_title="Data Preparation Stage",
        yaxis_title="Accuracy",
        height=500,
        hovermode='x unified',
        yaxis=dict(tickformat=".0%", range=[0.45, 0.55])  # Adjusted range for your data
    )
    
    st.plotly_chart(fig_timeline, width='stretch')
    
st.markdown("---")


# Data Preparation Evolution Timeline
# st.subheader("📅 Data Preparation Evolution & Impact")

st.markdown("#### Model Comparison by Metric")

# Comparison chart
if 'rf_results' in data and 'gbm_results' in data:
    # st.subheader("Model Comparison by Metric")
    
    # Create metric toggle
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            ["Random Forest", "GBM", "Both Models"],
            index=2  # Default to "Both Models"
        )
    
    with col2:
        # Determine available metrics based on selected model
        if selected_model == "Random Forest":
            metric_options = ["Accuracy", "F1"]
        elif selected_model == "GBM":
            metric_options = ["Accuracy", "F1", "AUC"]
        else:  # Both Models
            metric_options = ["Accuracy", "F1"]
        
        selected_metric = st.selectbox(
            "Select Metric",
            metric_options,
            index=0  # Default to Accuracy
        )
    
    with col3:
        # Add a toggle for text display
        show_values = st.toggle("Show Values", value=True)
    
    # Merge results
    rf = data['rf_results'].copy()
    gbm = data['gbm_results'].copy()
    
    rf['Model'] = 'Random Forest'
    gbm['Model'] = 'GBM'
    
    # Handle AUC column for RF if needed
    if selected_metric == "AUC" and 'AUC' not in rf.columns:
        rf['AUC'] = rf['Accuracy']  # Use accuracy as proxy for RF
    
    # Select data based on model choice
    if selected_model == "Random Forest":
        # Only show RF results
        if selected_metric == "Accuracy":
            rf_data = rf[['Scenario', 'Accuracy', 'Model']].rename(columns={'Accuracy': 'Value'})
            gbm_data = pd.DataFrame()  # Empty
        elif selected_metric == "F1":
            rf_data = rf[['Scenario', 'F1', 'Model']].rename(columns={'F1': 'Value'})
            gbm_data = pd.DataFrame()  # Empty
        elif selected_metric == "AUC":
            rf_data = rf[['Scenario', 'AUC', 'Model']].rename(columns={'AUC': 'Value'})
            gbm_data = pd.DataFrame()  # Empty
        
        combined = rf_data
        
    elif selected_model == "GBM":
        # Only show GBM results
        if selected_metric == "Accuracy":
            gbm_data = gbm[['Scenario', 'Accuracy', 'Model']].rename(columns={'Accuracy': 'Value'})
            rf_data = pd.DataFrame()  # Empty
        elif selected_metric == "F1":
            gbm_data = gbm[['Scenario', 'F1', 'Model']].rename(columns={'F1': 'Value'})
            rf_data = pd.DataFrame()  # Empty
        elif selected_metric == "AUC":
            gbm_data = gbm[['Scenario', 'AUC', 'Model']].rename(columns={'AUC': 'Value'})
            rf_data = pd.DataFrame()  # Empty
        
        combined = gbm_data
        
    else:  # Both Models
        # Show both models, but only common metrics (Accuracy or F1)
        if selected_metric == "Accuracy":
            rf_data = rf[['Scenario', 'Accuracy', 'Model']].rename(columns={'Accuracy': 'Value'})
            gbm_data = gbm[['Scenario', 'Accuracy', 'Model']].rename(columns={'Accuracy': 'Value'})
        else:  # F1
            rf_data = rf[['Scenario', 'F1', 'Model']].rename(columns={'F1': 'Value'})
            gbm_data = gbm[['Scenario', 'F1', 'Model']].rename(columns={'F1': 'Value'})
        
        combined = pd.concat([rf_data, gbm_data])
    
    # Create visualization
    if not combined.empty:
        # Format metric name for display
        metric_display_names = {
            'Accuracy': 'Accuracy',
            'F1': 'F1 Score',
            'AUC': 'AUC Score'
        }
        
        metric_display = metric_display_names.get(selected_metric, selected_metric)
        
        # Determine y-axis format
        if selected_metric == "AUC":
            y_format = ".3f"
            y_range = [0.4, 0.6]  # Adjust based on your AUC range
        else:
            y_format = ".2%"
            y_range = [0.45, 0.55]  # Adjust based on your data range
        
        # Create chart
        fig = px.bar(
            combined,
            x='Scenario',
            y='Value',
            color='Model',
            barmode='group' if selected_model == "Both Models" else 'group',
            title=f'{selected_model}: {metric_display} by Scenario',
            text='Value' if show_values else None,
            color_discrete_map={'Random Forest': "#636EFA", 'GBM': '#EF553B'},
            category_orders={"Model": ["Random Forest", "GBM"]}  # Ensure consistent order
        )
        
        # Update text formatting
        if show_values:
            if selected_metric == "AUC":
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            else:
                fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        
        # Update layout
        fig.update_layout(
            height=500,
            yaxis=dict(
                title=metric_display,
                tickformat=y_format,
                range=y_range
            ),
            hovermode='x unified',
            showlegend=selected_model == "Both Models" or len(combined['Model'].unique()) > 1
        )
        
        # Add horizontal line at baseline (0.5 for binary classification)
        if selected_metric in ["Accuracy", "F1"]:
            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                annotation_text="Baseline (Random)",
                annotation_position="bottom right"
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        with st.expander("📋 View Data Table"):
            # Pivot for better readability
            if selected_model == "Both Models":
                pivot_df = combined.pivot(index='Scenario', columns='Model', values='Value')
                if selected_metric == "Accuracy":
                    pivot_df = pivot_df.applymap(lambda x: f"{x:.2%}")
                elif selected_metric == "F1":
                    pivot_df = pivot_df.applymap(lambda x: f"{x:.3f}")
                elif selected_metric == "AUC":
                    pivot_df = pivot_df.applymap(lambda x: f"{x:.3f}")
                st.dataframe(pivot_df, use_container_width=True)
            else:
                display_df = combined[['Scenario', 'Value']].copy()
                if selected_metric == "Accuracy":
                    display_df['Value'] = display_df['Value'].apply(lambda x: f"{x:.2%}")
                elif selected_metric == "F1":
                    display_df['Value'] = display_df['Value'].apply(lambda x: f"{x:.3f}")
                elif selected_metric == "AUC":
                    display_df['Value'] = display_df['Value'].apply(lambda x: f"{x:.3f}")
                st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("No data available for the selected combination.")

st.markdown("---")