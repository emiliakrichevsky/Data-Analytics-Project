"""
Run H2O AutoML to find the best model for the dataset.

This script:
1.  Prepares the data using the "All Combined" strategy (Smart Imputation + Outlier Removal + Scaling).
2.  Initializes H2O.
3.  Runs AutoML for a maximum of 10 models (or 10 minutes).
4.  Outputs the leaderboard and saves the best model.
"""

import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
import os

# Import our refactored modules
import prepare_data
import create_features_and_labels
import detect_outliers
import scale_features

def prepare_best_dataset():
    """
    Prepare the dataset using the best strategy found in comparison:
    Smart Imputation -> Outlier Removal -> Scaling -> Feature Engineering
    """
    print("=" * 70)
    print("PREPARING DATA FOR AUTOML")
    print("=" * 70)
    
    # 1. Smart Imputation
    print("\n1. Smart Imputation...")
    df_smart = prepare_data.prepare_data()
    
    # 2. Outlier Removal
    print("\n2. Outlier Removal...")
    zscore_results, _ = detect_outliers.analyze_outliers(df_smart)
    df_no_outliers = detect_outliers.remove_outliers(df_smart, zscore_results, method='zscore')
    print(f"   Removed {len(df_smart) - len(df_no_outliers)} outlier rows")
    
    # 3. Scaling
    print("\n3. Scaling...")
    df_scaled_raw, _ = scale_features.process_scaling(df_no_outliers)
    
    # 4. Feature Engineering
    print("\n4. Feature Engineering...")
    df_ml, _ = create_features_and_labels.process_data(df_scaled_raw)
    
    # Drop rows with nulls (just in case any remain after engineering)
    df_ml = df_ml.dropna()
    
    # Drop non-feature columns for H2O
    # H2O needs the target 'y', but we should drop identifiers if we don't want them used
    # We'll keep them for now but specify predictors explicitly
    
    return df_ml

def main():
    # Initialize H2O
    print("\n" + "=" * 70)
    print("INITIALIZING H2O")
    print("=" * 70)
    h2o.init()
    
    # Prepare data
    df_ml = prepare_best_dataset()
    
    print("\n" + "=" * 70)
    print("SORTING DATA CHRONOLOGICALLY")
    print("=" * 70)
    
    # CRITICAL: Sort by Date FIRST (not by stock) to ensure true time series split
    df_ml = df_ml.sort_values('Date').reset_index(drop=True)
    print(f"  Date range: {df_ml['Date'].min().date()} to {df_ml['Date'].max().date()}")
    print(f"  Total rows: {len(df_ml)}")
    
    print("\n" + "=" * 70)
    print("CONVERTING TO H2O FRAME")
    print("=" * 70)
    
    # Convert to H2O Frame
    hf = h2o.H2OFrame(df_ml)
    
    # Define predictors and response
    y = 'y'
    ignore_cols = ['Date', 'stock', 'stock_fwd_ret_21d', 'sp500_fwd_ret_21d']
    x = [col for col in hf.columns if col not in ignore_cols and col != y]
    
    # TIME SERIES SPLIT: Use a strict DATE cutoff (not row percentage)
    # This prevents leakage when stocks have different date ranges
    print("\nCreating strict chronological split...")
    
    # Find the date that splits 80/20
    all_dates = sorted(df_ml['Date'].unique())
    split_date_idx = int(len(all_dates) * 0.8)
    split_date = all_dates[split_date_idx]
    
    print(f"  Split date: {split_date.date()}")
    
    # Split in pandas BEFORE converting to H2O
    train_df = df_ml[df_ml['Date'] < split_date].copy()
    valid_df = df_ml[df_ml['Date'] >= split_date].copy()
    
    print(f"  Training rows: {len(train_df)} (up to {train_df['Date'].max().date()})")
    print(f"  Validation rows: {len(valid_df)} (from {valid_df['Date'].min().date()} to {valid_df['Date'].max().date()})")
    
    # Convert to H2O
    train = h2o.H2OFrame(train_df)
    valid = h2o.H2OFrame(valid_df)
    
    # Convert target to factor
    train[y] = train[y].asfactor()
    valid[y] = valid[y].asfactor()
    
    # Run AutoML with time series split
    print("\n" + "=" * 70)
    print("RUNNING H2O AUTOML (TIME SERIES SPLIT - DATE CUTOFF)")
    print("=" * 70)
    
    aml = H2OAutoML(max_models=10, seed=42, max_runtime_secs=600, verbosity='info', nfolds=0)
    aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
    
    # View Leaderboard
    print("\n" + "=" * 70)
    print("LEADERBOARD")
    print("=" * 70)
    lb = aml.leaderboard
    print(lb.head(rows=lb.nrows))
    
    # Save Leaderboard
    lb_df = lb.as_data_frame()
    lb_df.to_csv('data/automl_leaderboard.csv', index=False)
    print("\nLeaderboard saved to data/automl_leaderboard.csv")
    
    # Best Model
    print("\n" + "=" * 70)
    print("BEST MODEL DETAILS")
    print("=" * 70)
    best_model = aml.leader
    print(best_model)
    
    # Save Best Model
    model_path = h2o.save_model(model=best_model, path="data/models", force=True)
    print(f"\nBest model saved to: {model_path}")

if __name__ == '__main__':
    main()
