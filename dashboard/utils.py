"""
Utility functions for the Stock Prediction Dashboard
"""

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.graph_objects as go
import plotly.express as px

@st.cache_data
def load_data():
    """Load all datasets with caching"""
    data = {}

    # Raw data
    if os.path.exists("../data/integrated_raw_data.csv"):
        data["raw"] = pd.read_csv("../data/integrated_raw_data.csv")
        data["raw"]["Date"] = pd.to_datetime(data["raw"]["Date"])

    # Prepared data
    if os.path.exists("../data/integrated_prepared_data.csv"):
        data["prepared"] = pd.read_csv("../data/integrated_prepared_data.csv")
        data["prepared"]["Date"] = pd.to_datetime(data["prepared"]["Date"])

    # ML dataset
    if os.path.exists("../data/ml_features_and_labels_clean.csv"):
        data["ml"] = pd.read_csv("../data/ml_features_and_labels_clean.csv")
        data["ml"]["Date"] = pd.to_datetime(data["ml"]["Date"])

    # Model results
    if os.path.exists("../data/model_comparison_timeseries_results.csv"):
        data["rf_results"] = pd.read_csv(
            "../data/model_comparison_timeseries_results.csv"
        )

    if os.path.exists("../data/model_comparison_timeseries_gbm_results.csv"):
        data["gbm_results"] = pd.read_csv(
            "../data/model_comparison_timeseries_gbm_results.csv"
        )

    # Outlier analysis
    if os.path.exists("../data/outlier_detection_summary.csv"):
        data["outliers"] = pd.read_csv("../data/outlier_detection_summary.csv")

    # Scaling log
    if os.path.exists("../data/scaling_log.csv"):
        data["scaling"] = pd.read_csv("../data/scaling_log.csv")

    return data


def get_data():
    """Get loaded data with session state caching"""
    if "data" not in st.session_state:
        st.session_state.data = load_data()
    return st.session_state.data


def create_page_header(title):
    """Create a consistent page header"""
    st.header(title)
    st.markdown("---")


def get_data_distribution(data, column):
    """Generate distribution statistics for a column"""
    return {
        "mean": data[column].mean(),
        "median": data[column].median(),
        "std": data[column].std(),
        "min": data[column].min(),
        "max": data[column].max(),
        "q25": data[column].quantile(0.25),
        "q75": data[column].quantile(0.75),
    }

# TODO: Refactor plotting functions to use Plotly graph objects
def plot_data_distribution(data, column, bins=30):
    """Plot histogram and KDE for data distribution"""
    # fig, ax = plt.subplots(figsize=(10, 5))
    # px.histogram(data[column].dropna(), nbins=bins)
    # ax.set_xlabel(column)
    # ax.set_ylabel("Frequency")
    # ax.set_title(f"Distribution of {column}")
    # fig_plotly = tls.mpl_to_plotly(fig)
    # return fig
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=data[column].dropna(),
            nbinsx=bins,
            histnorm="probability density",
            name="Histogram",
            marker_color="lightblue",
            opacity=0.75,
        )
    )
    return fig


def extract_feature_importance(results_df, top_n=10):
    """Extract feature importance from model results"""
    if "feature_importance" in results_df.columns:
        importance = results_df["feature_importance"].value_counts().head(top_n)
        return importance
    return None


def plot_feature_importance(importance_data, title="Feature Importance", top_n=10):
    """Plot feature importance as horizontal bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_data.head(top_n).plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.invert_yaxis()
    fig_plotly = tls.mpl_to_plotly(fig)
    return fig_plotly

def create_box_plot(data, column):
    """Create a box plot for a given column"""
    df = data.copy().dropna(subset=[column])
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            name="Data",
            x=df[column],
            boxpoints=False,
            marker=dict(color="rgb(8,81,156)"),
            line=dict(color="rgb(8,81,156)"),
        )
    )
    fig.update_layout(title=f"Box Plot of: {column}", xaxis_title=column)
    # fig = px.box(df, y=column, title=f'Box Plot of {column}')
    # fig_plotly = tls.mpl_to_plotly(fig)
    return fig

def create_outlier_box_plot(data, column):
    """Create a box plot for a given column"""
    df = data.copy().dropna(subset=[column])
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            name="Suspected Outliers",
            x=df[column],
            boxpoints="suspectedoutliers",
            marker=dict(
                color="rgb(8,81,156)",
                outliercolor="rgba(219, 64, 82, 0.6)",
                line=dict(outliercolor="rgba(219, 64, 82, 0.6)", outlierwidth=2)),
            
        )
    )
    fig.add_trace(
        go.Box(
            name="Outliers",
            x=df[column],
            boxpoints="outliers",
            marker=dict(color="rgb(107,174,214)"),
            line=dict(color="rgb(107,174,214)"),
        )
    )

    fig.update_layout(title=f"Box Plot of: {column}", xaxis_title=column)
    return fig
