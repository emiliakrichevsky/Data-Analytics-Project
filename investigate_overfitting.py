"""
Investigate Overfitting: Random Split vs Time Series Split

This script tests the hypothesis that the high model accuracy (90%+) is due to
data leakage caused by random splitting of time-series data.

It runs two experiments using the "All Combined" dataset:
1.  Random Split (Current approach)
2.  Time Series Split (Strict separation by date)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import prepare_data
import detect_outliers
import scale_features
import create_features_and_labels

def get_data():
    print("Preparing data (All Combined strategy)...")
    # 1. Smart Imputation
    df = prepare_data.prepare_data()
    # 2. Outlier Removal
    zscore_results, _ = detect_outliers.analyze_outliers(df)
    df = detect_outliers.remove_outliers(df, zscore_results, method='zscore')
    # 3. Scaling
    df, _ = scale_features.process_scaling(df)
    # 4. Feature Engineering
    df, _ = create_features_and_labels.process_data(df)
    # Drop nulls
    df = df.dropna()
    return df

def train_evaluate(X_train, X_test, y_train, y_test, name):
    print(f"\nEvaluating: {name}")
    print(f"  Train size: {len(X_train):,}")
    print(f"  Test size:  {len(X_test):,}")
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    return acc

def main():
    df = get_data()
    
    # Prepare X and y
    drop_cols = ['Date', 'stock', 'y', 'stock_fwd_ret_21d', 'sp500_fwd_ret_21d']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['y']
    dates = df['Date']
    
    print("\n" + "="*60)
    print("EXPERIMENT 1: RANDOM SPLIT (Current Approach)")
    print("="*60)
    # This shuffles data, mixing future and past
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)
    acc_random = train_evaluate(X_train_r, X_test_r, y_train_r, y_test_r, "Random Split")
    
    print("\n" + "="*60)
    print("EXPERIMENT 2: TIME SERIES SPLIT (Strict)")
    print("="*60)
    # Split by date: Train on first 80%, Test on last 20%
    split_idx = int(len(df) * 0.8)
    split_date = dates.iloc[split_idx]
    
    print(f"Splitting at date: {split_date}")
    
    X_train_t = X.iloc[:split_idx]
    X_test_t = X.iloc[split_idx:]
    y_train_t = y.iloc[:split_idx]
    y_test_t = y.iloc[split_idx:]
    
    acc_time = train_evaluate(X_train_t, X_test_t, y_train_t, y_test_t, "Time Series Split")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print(f"Random Split Accuracy:     {acc_random:.4f}")
    print(f"Time Series Split Accuracy:{acc_time:.4f}")
    print(f"Drop in Performance:       {(acc_random - acc_time)*100:.2f}%")
    
    if acc_time < 0.60:
        print("\nVERDICT: SEVERE OVERFITTING CONFIRMED.")
        print("The high accuracy was due to data leakage (looking ahead) caused by random shuffling.")
    elif acc_time < 0.80:
        print("\nVERDICT: MODERATE OVERFITTING.")
        print("The model learned some real patterns but relied heavily on autocorrelation.")
    else:
        print("\nVERDICT: MODEL IS ROBUST.")
        print("The model generalizes well even to the future.")

if __name__ == '__main__':
    main()
