import h2o
from h2o.estimators import H2OGradientBoostingEstimator
import pandas as pd
import prepare_data
import create_features_and_labels
import detect_outliers
import scale_features
import os

def prepare_best_dataset():
    print("Preparing data...")
    df_smart = prepare_data.prepare_data()
    zscore_results, _ = detect_outliers.analyze_outliers(df_smart)
    df_no_outliers = detect_outliers.remove_outliers(df_smart, zscore_results, method='zscore')
    df_scaled_raw, _ = scale_features.process_scaling(df_no_outliers)
    df_ml, _ = create_features_and_labels.process_data(df_scaled_raw)
    df_ml = df_ml.dropna()
    return df_ml

def main():
    h2o.init()
    
    # 1. Prepare Data
    df_ml = prepare_best_dataset()
    
    # 2. Split Data (Strict Time Series)
    df_ml = df_ml.sort_values('Date').reset_index(drop=True)
    all_dates = sorted(df_ml['Date'].unique())
    split_date_idx = int(len(all_dates) * 0.8)
    split_date = all_dates[split_date_idx]
    
    print(f"\nSplit Date: {split_date.date()}")
    
    train_df = df_ml[df_ml['Date'] < split_date].copy()
    valid_df = df_ml[df_ml['Date'] >= split_date].copy()
    
    train_hf = h2o.H2OFrame(train_df)
    valid_hf = h2o.H2OFrame(valid_df)
    
    # 3. Define Features
    y = 'y'
    ignore_cols = ['Date', 'stock', 'stock_fwd_ret_21d', 'sp500_fwd_ret_21d']
    x = [col for col in train_hf.columns if col not in ignore_cols and col != y]
    
    # Convert target to factor
    train_hf[y] = train_hf[y].asfactor()
    valid_hf[y] = valid_hf[y].asfactor()
    
    # 4. Train GBM
    print("\nTraining GBM on Train, Evaluating on Valid...")
    gbm = H2OGradientBoostingEstimator(ntrees=50, seed=42)
    gbm.train(x=x, y=y, training_frame=train_hf, validation_frame=valid_hf)
    
    # 5. Results
    print("\n" + "=" * 50)
    print("GBM PERFORMANCE")
    print("=" * 50)
    print(f"Train AUC: {gbm.auc(train=True):.4f}")
    print(f"Valid AUC: {gbm.auc(valid=True):.4f}")
    
    if gbm.auc(valid=True) > 0.9:
        print("\n⚠️  LEAKAGE CONFIRMED (AUC > 0.9)")
        print("Top Features:")
        print(gbm.varimp(use_pandas=True).head(10))
    else:
        print("\n✅ NO LEAKAGE (AUC is realistic)")

if __name__ == "__main__":
    main()
