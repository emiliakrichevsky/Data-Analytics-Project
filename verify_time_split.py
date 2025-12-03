"""
Verify Time Series Split in H2O AutoML

This script checks if the H2O AutoML train/validation split is truly chronological.
"""

import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np

# Import our refactored modules
import prepare_data
import create_features_and_labels
import detect_outliers
import scale_features

def prepare_best_dataset():
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
    
    # Drop rows with nulls
    df_ml = df_ml.dropna()
    
    return df_ml

def main():
    # Prepare data
    df_ml = prepare_best_dataset()
    
    print("\n" + "=" * 70)
    print("VERIFYING TIME SERIES SPLIT")
    print("=" * 70)
    
    # Check if data is sorted
    print("\nChecking data order...")
    print(f"First 5 rows (Date, Stock):")
    print(df_ml[['Date', 'stock']].head())
    print(f"\nLast 5 rows (Date, Stock):")
    print(df_ml[['Date', 'stock']].tail())
    
    # Check date ranges by stock
    print("\nDate ranges by stock:")
    for stock in df_ml['stock'].unique():
        stock_df = df_ml[df_ml['stock'] == stock]
        print(f"  {stock}: {stock_df['Date'].min().date()} to {stock_df['Date'].max().date()} ({len(stock_df)} rows)")
    
    # Simulate the H2O split
    split_idx = int(len(df_ml) * 0.8)
    train_df = df_ml.iloc[:split_idx]
    valid_df = df_ml.iloc[split_idx:]
    
    print(f"\nSplit at index {split_idx}:")
    print(f"  Training rows: {len(train_df)}")
    print(f"  Validation rows: {len(valid_df)}")
    
    print(f"\nTraining set:")
    print(f"  Date range: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
    print(f"  Stocks: {train_df['stock'].value_counts().to_dict()}")
    
    print(f"\nValidation set:")
    print(f"  Date range: {valid_df['Date'].min().date()} to {valid_df['Date'].max().date()}")
    print(f"  Stocks: {valid_df['stock'].value_counts().to_dict()}")
    
    # Check for date overlap
    train_max = train_df['Date'].max()
    valid_min = valid_df['Date'].min()
    
    print(f"\n" + "=" * 70)
    print("LEAKAGE CHECK")
    print("=" * 70)
    if valid_min < train_max:
        print(f"⚠️  WARNING: DATA LEAKAGE DETECTED!")
        print(f"  Training set includes data up to: {train_max.date()}")
        print(f"  Validation set starts from: {valid_min.date()}")
        print(f"  Overlap: Validation has data from BEFORE the last training date")
        print(f"\n  This means the model can 'peek' at the past to predict the 'future'")
    else:
        print(f"✓ No leakage: Training ends at {train_max.date()}, Validation starts at {valid_min.date()}")

if __name__ == '__main__':
    main()
