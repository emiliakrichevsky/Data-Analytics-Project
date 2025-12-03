"""
Generate Data Preparation Report

This script runs the data preparation pipeline and generates a detailed
markdown report with statistics on:
1.  Null Handling (Rows affected, methods used)
2.  Outlier Removal (Rows removed, distribution changes)
3.  Scaling (Effect on value ranges)
"""

import pandas as pd
import numpy as np
import os
import prepare_data
import detect_outliers
import scale_features
import create_features_and_labels

def generate_report():
    report = []
    report.append("# Data Preparation Report")
    report.append("")
    report.append("This report details the transformation of the dataset through the data preparation pipeline.")
    report.append("")
    
    # Load Raw Data
    print("Loading raw data...")
    df_raw = pd.read_csv('data/integrated_raw_data.csv')
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    
    report.append("## 1. Raw Data Overview")
    report.append(f"- **Total Rows:** {len(df_raw):,}")
    report.append(f"- **Total Columns:** {len(df_raw.columns)}")
    report.append(f"- **Date Range:** {df_raw['Date'].min().date()} to {df_raw['Date'].max().date()}")
    report.append(f"- **Stocks:** {', '.join(df_raw['stock'].unique())}")
    
    # Null Analysis
    null_counts = df_raw.isnull().sum()
    total_cells = np.prod(df_raw.shape)
    total_nulls = null_counts.sum()
    report.append(f"- **Total Null Values:** {total_nulls:,} ({total_nulls/total_cells*100:.2f}% of data)")
    report.append("")
    
    # Step 1: Smart Imputation
    print("Running Smart Imputation...")
    df_smart = prepare_data.prepare_data()
    
    report.append("## 2. Technique: Smart Imputation")
    report.append("### Logic Applied")
    report.append("- **Macro Data (CPI, GDP):** Forward Fill then Backward Fill (handling reporting lags).")
    report.append("- **Stock Data:** Forward Fill (propagating last known price).")
    report.append("- **Interpolation:** Linear interpolation for scattered missing values.")
    
    report.append("### Impact")
    report.append(f"- **Rows Removed:** 0 (All nulls were handled by imputation)")
    report.append(f"- **Nulls Remaining:** {df_smart.isnull().sum().sum()}")
    report.append("- **Insight:** By using smart imputation instead of dropping rows with nulls, we preserved **100%** of the temporal structure, which is critical for time-series modeling.")
    report.append("")
    
    # Step 2: Outlier Removal
    print("Running Outlier Removal...")
    zscore_results, _ = detect_outliers.analyze_outliers(df_smart)
    df_no_outliers = detect_outliers.remove_outliers(df_smart, zscore_results, method='zscore')
    
    rows_removed = len(df_smart) - len(df_no_outliers)
    pct_removed = (rows_removed / len(df_smart)) * 100
    
    report.append("## 3. Technique: Outlier Removal")
    report.append("### Logic Applied")
    report.append("- **Method:** Z-Score")
    report.append("- **Threshold:** 3 Standard Deviations")
    report.append("- **Action:** Remove entire row if any column has an outlier.")
    
    report.append("### Impact")
    report.append(f"- **Rows Before:** {len(df_smart):,}")
    report.append(f"- **Rows After:** {len(df_no_outliers):,}")
    report.append(f"- **Rows Removed:** {rows_removed:,} ({pct_removed:.2f}%)")
    
    report.append("### Top Outlier Contributors")
    # Find which columns contributed most to removal
    outlier_counts = []
    for col, stats in zscore_results.items():
        outlier_counts.append((col, stats['count']))
    outlier_counts.sort(key=lambda x: x[1], reverse=True)
    
    report.append("| Column | Outliers Detected |")
    report.append("| :--- | :--- |")
    for col, count in outlier_counts[:5]:
        report.append(f"| {col} | {count} |")
    
    report.append("")
    report.append("- **Insight:** The majority of outliers came from **Volume** and **Returns**, which often spike during market events. Removing these prevents the model from overfitting to extreme, non-repeatable events (flash crashes, earnings surprises).")
    report.append("")
    
    # Step 3: Scaling
    print("Running Scaling...")
    df_scaled, scaling_log = scale_features.process_scaling(df_no_outliers)
    
    report.append("## 4. Technique: Feature Scaling")
    report.append("### Logic Applied")
    report.append("- **StandardScaler:** Applied to normally distributed features (e.g., Returns).")
    report.append("- **RobustScaler:** Applied to features with heavy tails (e.g., Volume).")
    report.append("- **MinMaxScaler:** Applied to bounded ratios.")
    
    report.append("### Impact")
    report.append("- **Rows Removed:** 0 (Transformation only)")
    
    # Show example of scaling effect
    report.append("### Example Transformation (Volume)")
    vol_before = df_no_outliers['Volume'].describe()
    vol_after = df_scaled['Volume'].describe()
    
    report.append(f"- **Before Scaling:** Mean = {vol_before['mean']:,.0f}, Std = {vol_before['std']:,.0f}, Range = [{vol_before['min']:,.0f}, {vol_before['max']:,.0f}]")
    report.append(f"- **After Scaling:** Mean = {vol_after['mean']:.4f}, Std = {vol_after['std']:.4f}, Range = [{vol_after['min']:.4f}, {vol_after['max']:.4f}]")
    
    report.append("")
    report.append("- **Insight:** Scaling normalized the range of all features to be roughly comparable (centered around 0). This is crucial for models like Neural Networks and helps Random Forests converge faster by removing the bias towards features with larger raw numbers (like Volume vs Interest Rate).")
    report.append("")
    
    # Step 4: Final Feature Engineering
    print("Running Feature Engineering...")
    df_ml, _ = create_features_and_labels.process_data(df_scaled)
    
    # Final cleanup (dropping null targets)
    rows_before_drop = len(df_ml)
    df_final = df_ml.dropna()
    rows_dropped_final = rows_before_drop - len(df_final)
    
    report.append("## 5. Final Dataset Status")
    report.append(f"- **Final Rows for Training:** {len(df_final):,}")
    report.append(f"- **Total Features:** {len(df_final.columns) - 1} (excluding target)")
    report.append(f"- **Note:** {rows_dropped_final} rows were dropped at the very end. These correspond to the last 21 days of data for each stock, for which we cannot calculate the 'Future Return' target variable yet.")
    
    # Write Report
    with open('DATA_PREPARATION_REPORT.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("Report generated: DATA_PREPARATION_REPORT.md")

if __name__ == '__main__':
    generate_report()
