import streamlit as st
import time
import numpy as np
from utils import load_data
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(page_title="Model Results", page_icon=":material/smart_toy:", layout="wide")
st.markdown("# Model Results")
st.sidebar.header("Model Results")

data = load_data()
if not data or ('rf_results' not in data and 'gbm_results' not in data):
    st.warning("No model results available. Please train models first.")
else:
    st.subheader("Time-Series Split Results (Realistic)")
    col1, col2 = st.columns(2)
# Random Forest Results
    with col1:
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
            color_continuous_scale='viridis'
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')

# GBM Results
    with col2:
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
            color_continuous_scale='sunset'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Enhanced Model Performance Comparison
# 1. Performance Metrics Comparison (Accuracy, F1, AUC)
if 'rf_results' in data and 'gbm_results' in data:
    st.subheader("Impact of Data Preparation Techniques on Model Performance")
    
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
    # st.markdown("##### 📈 Metrics Radar Chart by Model")
    
    # Prepare data for radar chart
    # scenarios = ['Scaled Data', 'All Combined']  # Best scenarios
    # metrics = ['Accuracy', 'F1']
    
    # fig_radar = go.Figure()
    
    # for model in ['Random Forest', 'GBM']:
    #     for scenario in scenarios:
    #         model_scenario_data = df_comparison[
    #             (df_comparison['Model'] == model) & 
    #             (df_comparison['Scenario'] == scenario)
    #         ]
    #         if len(model_scenario_data) > 0:
    #             values = []
    #             for metric in metrics:
    #                 val = model_scenario_data[model_scenario_data['Metric'] == metric]['Value'].values
    #                 if len(val) > 0:
    #                     values.append(val[0])
    #                 else:
    #                     values.append(0)
                
    #             # Add scenario data
    #             fig_radar.add_trace(go.Scatterpolar(
    #                 r=values + [values[0]],  # Close the polygon
    #                 theta=metrics + [metrics[0]],
    #                 name=f"{model} - {scenario}",
    #                 fill='toself',
    #                 opacity=0.6
    #             ))
    
    # fig_radar.update_layout(
    #     polar=dict(
    #         radialaxis=dict(
    #             visible=True,
    #             range=[0.35, 0.65]  # Adjusted range based on your data
    #         )),
    #     showlegend=True,
    #     height=500,
    #     title="Performance Metrics Comparison (Best Scenarios)"
    # )
    
    # st.plotly_chart(fig_radar, width='stretch')
    
    # 2. Model Improvement Heatmap
    # st.markdown("##### 🔥 Metric: Accuracy")
    
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
            text_auto=True,
            aspect="auto",
            color_continuous_scale='YlGnBu',
            title="Accuracy Improvement Over Baseline Model (%)",
            labels=dict(x="Model", y="Data Prep Strategy", color="Improvement (%)")
        )
        fig_heatmap.update_traces(texttemplate='%{z:.2f}%', textfont={'size':12})
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, width='stretch')
