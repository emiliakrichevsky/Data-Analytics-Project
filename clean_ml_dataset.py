"""
Clean ML dataset by removing rows with null values
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("CLEANING ML DATASET - REMOVING NULL ROWS")
print("=" * 70)

# Load dataset
df = pd.read_csv('data/ml_features_and_labels.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['stock', 'Date']).reset_index(drop=True)

print(f"\nOriginal dataset: {len(df):,} rows")

# Step 1: Remove rows where y (label) is null (last 21 days - no future data)
print("\n1. Removing rows with null labels (y)...")
rows_before = len(df)
df_clean = df[df['y'].notna()].copy()
rows_removed_labels = rows_before - len(df_clean)
print(f"   Removed {rows_removed_labels} rows (last 21 days per stock)")

# Step 2: Remove rows where critical features are null
# Remove first rows that don't have rolling window features
print("\n2. Removing rows with null features...")
rows_before = len(df_clean)

# Keep only rows where all features are available
# Exclude forward returns columns (not needed after label creation)
feature_cols = [col for col in df_clean.columns 
                if col not in ['Date', 'stock', 'y', 'stock_fwd_ret_21d', 'sp500_fwd_ret_21d']]

# Remove rows where any feature is null
df_clean = df_clean.dropna(subset=feature_cols).copy()
rows_removed_features = rows_before - len(df_clean)
print(f"   Removed {rows_removed_features} rows (missing rolling window features)")

# Step 3: Reset index
df_clean = df_clean.reset_index(drop=True)

# Summary
print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)
print(f"\nOriginal rows: {len(df):,}")
print(f"Rows removed (null labels): {rows_removed_labels}")
print(f"Rows removed (null features): {rows_removed_features}")
print(f"Final clean rows: {len(df_clean):,}")
print(f"Rows removed total: {len(df) - len(df_clean)}")
print(f"Retention rate: {(len(df_clean) / len(df) * 100):.2f}%")

# Check final null counts
final_nulls = df_clean.isnull().sum().sum()
print(f"\nNull values in clean dataset: {final_nulls}")
if final_nulls > 0:
    print("Warning: Some nulls remain!")
    print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# Label distribution
print(f"\nLabel distribution:")
label_counts = df_clean['y'].value_counts()
print(f"  y=0 (underperform): {label_counts.get(0, 0):,}")
print(f"  y=1 (outperform): {label_counts.get(1, 0):,}")
print(f"  Outperform rate: {(label_counts.get(1, 0) / len(df_clean) * 100):.2f}%")

# Save clean dataset
print("\n3. Saving clean dataset...")
output_file = 'data/ml_features_and_labels_clean.csv'
df_clean.to_csv(output_file, index=False, float_format='%.6f')
print(f"   OK Saved to: {output_file}")

# Also save a version without forward return columns (not needed for training)
df_ml_ready = df_clean.drop(columns=['stock_fwd_ret_21d', 'sp500_fwd_ret_21d'])
output_file_ready = 'data/ml_dataset_ready.csv'
df_ml_ready.to_csv(output_file_ready, index=False, float_format='%.6f')
print(f"   OK Saved ML-ready dataset to: {output_file_ready}")

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

