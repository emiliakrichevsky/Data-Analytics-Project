"""
Compare Stacked Ensemble Accuracy with Different Data Preparation Techniques
Using TIME SERIES SPLIT (Realistic Evaluation)

This compares how Stacked Ensemble performs across 5 data prep scenarios.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os

# Import our refactored modules
import prepare_data
import create_features_and_labels
import detect_outliers
import scale_features

def create_stacked_ensemble():
    """
    Create a Stacked Ensemble similar to H2O's approach.
    Uses RF, GBM as base learners with Logistic Regression as meta-learner.
    """
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('gbm', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3,  # Internal cross-validation for meta-learner
        n_jobs=-1
    )
    
    return stack

def train_and_evaluate_timeseries(df, name="Model"):
    """
    Train a Stacked Ensemble using TIME SERIES SPLIT and evaluate.
    """
    print(f"\nTraining {name}...")
    
    # Drop rows with nulls
    df_clean = df.dropna()
    
    if len(df_clean) == 0:
        print("  Error: No data remaining after dropping nulls!")
        return 0, 0, 0
    
    # Sort by date
    df_clean = df_clean.sort_values('Date').reset_index(drop=True)
    
    # Separate features and target
    drop_cols = ['Date', 'stock', 'y', 'stock_fwd_ret_21d', 'sp500_fwd_ret_21d']
    X = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])
    y = df_clean['y']
    
    # TIME SERIES SPLIT: Use date cutoff
    all_dates = sorted(df_clean['Date'].unique())
    split_date_idx = int(len(all_dates) * 0.8)
    split_date = all_dates[split_date_idx]
    
    train_mask = df_clean['Date'] < split_date
    test_mask = df_clean['Date'] >= split_date
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"  Split date: {split_date.date()}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Train Stacked Ensemble
    ensemble = create_stacked_ensemble()
    ensemble.fit(X_train, y_train)
    
    # Predict
    y_pred = ensemble.predict(X_test)
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    return acc, f1, auc

def main():
    print("=" * 70)
    print("STACKED ENSEMBLE COMPARISON - TIME SERIES SPLIT")
    print("=" * 70)
    
    results = []
    
    # Load raw integrated data
    print("\nLoading raw data...")
    df_raw = pd.read_csv('data/integrated_raw_data.csv')
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    
    # ========== SCENARIO 1: BASELINE (Minimal Prep) ==========
    print("\n" + "-" * 70)
    print("SCENARIO 1: BASELINE (Drop Nulls, No Scaling, No Outlier Removal)")
    print("-" * 70)
    
    df_baseline = df_raw.copy()
    df_baseline = df_baseline.ffill().dropna()
    
    print("Engineering features for baseline...")
    df_baseline_ml, _ = create_features_and_labels.process_data(df_baseline)
    
    acc_1, f1_1, auc_1 = train_and_evaluate_timeseries(df_baseline_ml, "Baseline Ensemble")
    results.append({'Scenario': 'Baseline (Simple Fill)', 'Accuracy': acc_1, 'F1': f1_1, 'AUC': auc_1})
    
    # ========== SCENARIO 2: SMART NULL HANDLING ==========
    print("\n" + "-" * 70)
    print("SCENARIO 2: SMART NULL HANDLING (Imputation)")
    print("-" * 70)
    
    print("Applying smart null handling...")
    df_smart = prepare_data.prepare_data()
    
    print("Engineering features...")
    df_smart_ml, _ = create_features_and_labels.process_data(df_smart)
    
    acc_2, f1_2, auc_2 = train_and_evaluate_timeseries(df_smart_ml, "Smart Imputation Ensemble")
    results.append({'Scenario': 'Smart Imputation', 'Accuracy': acc_2, 'F1': f1_2, 'AUC': auc_2})
    
    # ========== SCENARIO 3: OUTLIER REMOVAL ==========
    print("\n" + "-" * 70)
    print("SCENARIO 3: OUTLIER REMOVAL (on top of Smart Imputation)")
    print("-" * 70)
    
    print("Detecting and removing outliers...")
    zscore_results, iqr_results = detect_outliers.analyze_outliers(df_smart)
    df_no_outliers = detect_outliers.remove_outliers(df_smart, zscore_results, method='zscore')
    
    print(f"Removed {len(df_smart) - len(df_no_outliers)} outlier rows")
    
    print("Engineering features...")
    df_outlier_ml, _ = create_features_and_labels.process_data(df_no_outliers)
    
    acc_3, f1_3, auc_3 = train_and_evaluate_timeseries(df_outlier_ml, "Outlier Removal Ensemble")
    results.append({'Scenario': 'Outlier Removal', 'Accuracy': acc_3, 'F1': f1_3, 'AUC': auc_3})
    
    # ========== SCENARIO 4: SCALING ==========
    print("\n" + "-" * 70)
    print("SCENARIO 4: SCALING (on top of Smart Imputation)")
    print("-" * 70)
    
    print("Scaling raw features...")
    df_scaled_raw, _ = scale_features.process_scaling(df_smart)
    
    print("Engineering features from scaled data...")
    df_scaled_ml, _ = create_features_and_labels.process_data(df_scaled_raw)
    
    acc_4, f1_4, auc_4 = train_and_evaluate_timeseries(df_scaled_ml, "Scaled Data Ensemble")
    results.append({'Scenario': 'Scaled Data', 'Accuracy': acc_4, 'F1': f1_4, 'AUC': auc_4})
    
    # ========== SCENARIO 5: ALL COMBINED ==========
    print("\n" + "-" * 70)
    print("SCENARIO 5: ALL COMBINED (Smart + Outliers + Scaling)")
    print("-" * 70)
    
    print("Scaling outlier-free data...")
    df_final_raw, _ = scale_features.process_scaling(df_no_outliers)
    
    print("Engineering features...")
    df_final_ml, _ = create_features_and_labels.process_data(df_final_raw)
    
    acc_5, f1_5, auc_5 = train_and_evaluate_timeseries(df_final_ml, "Combined Ensemble")
    results.append({'Scenario': 'All Combined', 'Accuracy': acc_5, 'F1': f1_5, 'AUC': auc_5})
    
    # ========== RESULTS ==========
    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON (STACKED ENSEMBLE - TIME SERIES SPLIT)")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('data/model_comparison_timeseries_ensemble_results.csv', index=False)
    print("\nResults saved to data/model_comparison_timeseries_ensemble_results.csv")

if __name__ == '__main__':
    main()
