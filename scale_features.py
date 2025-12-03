"""
Feature Scaling/Normalization Script

Applies appropriate scaling methods to the prepared dataset.
Documents all transformations applied.
"""

import pandas as pd
import numpy as np
import os

# Manual implementation of scalers (no sklearn dependency)
def standard_scale(series):
    """StandardScaler: (x - mean) / std"""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return series
    return (series - mean) / std

def robust_scale(series):
    """RobustScaler: (x - median) / IQR"""
    median = series.median()
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    if iqr == 0:
        return series - median
    return (series - median) / iqr

def minmax_scale(series):
    """MinMaxScaler: (x - min) / (max - min)"""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0, index=series.index)
    return (series - min_val) / (max_val - min_val)

def analyze_data_for_scaling(df):
    """
    Analyze data to determine appropriate scaling methods.
    """
    print("=" * 70)
    print("DATA ANALYSIS FOR SCALING")
    print("=" * 70)
    
    # Identify columns to exclude from scaling
    exclude_cols = ['Date', 'stock']
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"\nTotal numeric columns: {len(numeric_cols)}")
    print(f"Columns to exclude: {exclude_cols}")
    print(f"Columns to scale: {len(numeric_cols)}")
    
    # Analyze each column
    scaling_analysis = {}
    
    print("\n" + "-" * 70)
    print("COLUMN ANALYSIS:")
    print("-" * 70)
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        mean_val = col_data.mean()
        std_val = col_data.std()
        median_val = col_data.median()
        min_val = col_data.min()
        max_val = col_data.max()
        
        # Check for outliers (values beyond 3 std dev)
        outliers = len(col_data[(col_data < mean_val - 3*std_val) | (col_data > mean_val + 3*std_val)])
        outlier_pct = (outliers / len(col_data)) * 100
        
        # Determine appropriate scaler
        if col in ['spy_RSI', 'qqq_RSI']:
            scaler_type = 'No scaling'
            reason = 'RSI is already bounded (0-100)'
        elif col.endswith('_Ratio') or 'ratio' in col.lower():
            scaler_type = 'MinMaxScaler'
            reason = 'Ratios are naturally bounded'
        elif outlier_pct > 5:
            scaler_type = 'RobustScaler'
            reason = f'High outliers ({outlier_pct:.1f}% beyond 3 std)'
        elif std_val == 0:
            scaler_type = 'No scaling'
            reason = 'Constant values (std=0)'
        else:
            scaler_type = 'StandardScaler'
            reason = 'Normal distribution, low outliers'
        
        scaling_analysis[col] = {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'min': min_val,
            'max': max_val,
            'outlier_pct': outlier_pct,
            'scaler': scaler_type,
            'reason': reason
        }
    
    return scaling_analysis, numeric_cols

def apply_scaling(df, scaling_analysis, numeric_cols):
    """
    Apply scaling based on analysis.
    """
    print("\n" + "=" * 70)
    print("APPLYING SCALING TRANSFORMATIONS")
    print("=" * 70)
    
    df_scaled = df.copy()
    scaling_log = []
    
    # Group columns by scaler type
    scaler_groups = {
        'StandardScaler': [],
        'RobustScaler': [],
        'MinMaxScaler': [],
        'No scaling': []
    }
    
    for col in numeric_cols:
        if col in scaling_analysis:
            scaler_type = scaling_analysis[col]['scaler']
            scaler_groups[scaler_type].append(col)
    
    # Apply StandardScaler
    if scaler_groups['StandardScaler']:
        print(f"\n1. StandardScaler ({len(scaler_groups['StandardScaler'])} columns)")
        print("-" * 70)
        
        for col in scaler_groups['StandardScaler']:
            before_stats = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
            
            # Apply standard scaling
            df_scaled[col] = standard_scale(df[col])
            
            after_stats = {
                'mean': df_scaled[col].mean(),
                'std': df_scaled[col].std(),
                'min': df_scaled[col].min(),
                'max': df_scaled[col].max()
            }
            
            scaling_log.append({
                'column': col,
                'method': 'StandardScaler',
                'before_mean': before_stats['mean'],
                'before_std': before_stats['std'],
                'after_mean': after_stats['mean'],
                'after_std': after_stats['std'],
                'reason': scaling_analysis[col]['reason']
            })
            
            print(f"   {col}:")
            print(f"     Before: mean={before_stats['mean']:.4f}, std={before_stats['std']:.4f}")
            print(f"     After: mean~{after_stats['mean']:.4f}, std~{after_stats['std']:.4f}")
    
    # Apply RobustScaler
    if scaler_groups['RobustScaler']:
        print(f"\n2. RobustScaler ({len(scaler_groups['RobustScaler'])} columns)")
        print("-" * 70)
        
        for col in scaler_groups['RobustScaler']:
            before_stats = {
                'median': df[col].median(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75),
                'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
            }
            
            df_scaled[col] = robust_scale(df[col])
            
            after_stats = {
                'median': df_scaled[col].median(),
                'mean': df_scaled[col].mean(),
                'std': df_scaled[col].std()
            }
            
            scaling_log.append({
                'column': col,
                'method': 'RobustScaler',
                'before_median': before_stats['median'],
                'before_iqr': before_stats['iqr'],
                'after_median': after_stats['median'],
                'after_mean': after_stats['mean'],
                'after_std': after_stats['std'],
                'reason': scaling_analysis[col]['reason']
            })
            
            print(f"   {col}:")
            print(f"     Before: median={before_stats['median']:.4f}, IQR={before_stats['iqr']:.4f}")
            print(f"     After: median~{after_stats['median']:.4f}, mean~{after_stats['mean']:.4f}")
    
    # Apply MinMaxScaler (if needed)
    if scaler_groups['MinMaxScaler']:
        print(f"\n3. MinMaxScaler ({len(scaler_groups['MinMaxScaler'])} columns)")
        print("-" * 70)
        
        for col in scaler_groups['MinMaxScaler']:
            before_stats = {
                'min': df[col].min(),
                'max': df[col].max()
            }
            
            df_scaled[col] = minmax_scale(df[col])
            
            after_stats = {
                'min': df_scaled[col].min(),
                'max': df_scaled[col].max()
            }
            
            scaling_log.append({
                'column': col,
                'method': 'MinMaxScaler',
                'before_min': before_stats['min'],
                'before_max': before_stats['max'],
                'after_min': after_stats['min'],
                'after_max': after_stats['max'],
                'reason': scaling_analysis[col]['reason']
            })
            
            print(f"   {col}:")
            print(f"     Before: min={before_stats['min']:.4f}, max={before_stats['max']:.4f}")
            print(f"     After: min~{after_stats['min']:.4f}, max~{after_stats['max']:.4f}")
    
    # Columns not scaled
    if scaler_groups['No scaling']:
        print(f"\n4. Not Scaled ({len(scaler_groups['No scaling'])} columns)")
        print("-" * 70)
        for col in scaler_groups['No scaling']:
            print(f"   {col}: {scaling_analysis[col]['reason']}")
            scaling_log.append({
                'column': col,
                'method': 'No scaling',
                'reason': scaling_analysis[col]['reason']
            })
    
    return df_scaled, scaling_log

def create_scaling_report(df_original, df_scaled, scaling_log):
    """
    Create a comprehensive report of scaling transformations.
    """
    report = []
    report.append("=" * 70)
    report.append("FEATURE SCALING/NORMALIZATION REPORT")
    report.append("=" * 70)
    report.append("")
    report.append(f"Report generated from: integrated_prepared_data.csv")
    report.append(f"Total rows: {len(df_original):,}")
    report.append(f"Total columns: {len(df_original.columns)}")
    report.append("")
    
    report.append("=" * 70)
    report.append("OVERVIEW")
    report.append("=" * 70)
    report.append("")
    report.append("This report documents all feature scaling/normalization transformations")
    report.append("applied to prepare features for machine learning modeling.")
    report.append("")
    
    # Summary statistics
    report.append("=" * 70)
    report.append("SCALING SUMMARY")
    report.append("=" * 70)
    report.append("")
    
    method_counts = {}
    for entry in scaling_log:
        method = entry['method']
        method_counts[method] = method_counts.get(method, 0) + 1
    
    report.append(f"Methods Applied:")
    for method, count in method_counts.items():
        report.append(f"  {method}: {count} columns")
    report.append("")
    
    # Detailed transformations
    report.append("=" * 70)
    report.append("DETAILED TRANSFORMATIONS")
    report.append("=" * 70)
    report.append("")
    
    for entry in scaling_log:
        col = entry['column']
        method = entry['method']
        reason = entry['reason']
        
        report.append(f"Column: {col}")
        report.append(f"  Method: {method}")
        report.append(f"  Reason: {reason}")
        
        if method == 'StandardScaler':
            report.append(f"  Before: mean={entry['before_mean']:.4f}, std={entry['before_std']:.4f}")
            report.append(f"  After: mean~{entry['after_mean']:.4f}, std~{entry['after_std']:.4f}")
            report.append(f"  Effect: Transformed to mean=0, std=1 (standard normal distribution)")
        
        elif method == 'RobustScaler':
            report.append(f"  Before: median={entry['before_median']:.4f}, IQR={entry['before_iqr']:.4f}")
            report.append(f"  After: median~{entry['after_median']:.4f}, mean~{entry['after_mean']:.4f}")
            report.append(f"  Effect: Transformed using median and IQR (robust to outliers)")
        
        elif method == 'MinMaxScaler':
            report.append(f"  Before: min={entry['before_min']:.4f}, max={entry['before_max']:.4f}")
            report.append(f"  After: min~{entry['after_min']:.4f}, max~{entry['after_max']:.4f}")
            report.append(f"  Effect: Transformed to range [0, 1]")
        
        report.append("")
    
    # Methodology explanation
    report.append("=" * 70)
    report.append("METHODOLOGY")
    report.append("=" * 70)
    report.append("")
    report.append("1. StandardScaler:")
    report.append("   - Formula: (x - mean) / std")
    report.append("   - Result: Mean=0, Standard Deviation=1")
    report.append("   - Use for: Normally distributed data, low outliers")
    report.append("   - Examples: Returns, percentage changes, z-scores")
    report.append("")
    report.append("2. RobustScaler:")
    report.append("   - Formula: (x - median) / IQR")
    report.append("   - Result: Median=0, IQR-based scaling")
    report.append("   - Use for: Data with outliers, non-normal distributions")
    report.append("   - Examples: Volume, prices, macro indicators")
    report.append("")
    report.append("3. MinMaxScaler:")
    report.append("   - Formula: (x - min) / (max - min)")
    report.append("   - Result: Range [0, 1]")
    report.append("   - Use for: Bounded features, ratios")
    report.append("   - Examples: RSI (0-100), ratios")
    report.append("")
    report.append("4. No Scaling:")
    report.append("   - Applied to: Already normalized features, constant values")
    report.append("   - Examples: RSI (already 0-100), binary features")
    report.append("")
    
    # Column categories
    report.append("=" * 70)
    report.append("COLUMN CATEGORIES")
    report.append("=" * 70)
    report.append("")
    
    categories = {
        'Stock Price Features': ['Close', 'High', 'Low', 'Open', 'Volume'],
        'SP500 Features': ['sp500_Close', 'sp500_High', 'sp500_Low', 'sp500_Open', 'sp500_Volume'],
        'Macro Indicators': ['CPI', 'Fed_Funds_Rate', 'GDP', 'Unemployment_Rate'],
        'Market Sentiment': ['VIX', 'Put_Call_Ratio', 'Market_Breadth'],
        'Sector ETFs': ['sector_XLK', 'sector_XLF', 'sector_XLV', 'sector_XLE', 'sector_XLI'],
        'Technical Indicators': ['spy_RSI', 'spy_SMA_50', 'spy_SMA_200', 'qqq_RSI', 'qqq_SMA_50', 'qqq_SMA_200']
    }
    
    for category, cols in categories.items():
        report.append(f"{category}:")
        for col in cols:
            if col in [e['column'] for e in scaling_log]:
                method = next(e['method'] for e in scaling_log if e['column'] == col)
                report.append(f"  - {col}: {method}")
        report.append("")
    
    # Excluded columns
    report.append("=" * 70)
    report.append("EXCLUDED FROM SCALING")
    report.append("=" * 70)
    report.append("")
    report.append("The following columns were NOT scaled:")
    report.append("  - Date: Temporal identifier, not a feature")
    report.append("  - stock: Categorical variable (will be encoded separately)")
    report.append("")
    
    # Notes
    report.append("=" * 70)
    report.append("IMPORTANT NOTES")
    report.append("=" * 70)
    report.append("")
    report.append("1. Scaling is applied to entire dataset (train + test)")
    report.append("   - In production: Fit scaler on training data only")
    report.append("   - Transform both train and test with fitted scaler")
    report.append("")
    report.append("2. Grouped scaling by stock:")
    report.append("   - Not applied here, but consider for production")
    report.append("   - Prevents one stock's distribution from affecting another")
    report.append("")
    report.append("3. Scaling is essential for:")
    report.append("   - Linear models (Logistic Regression, SVM)")
    report.append("   - Distance-based algorithms (KNN)")
    report.append("   - Neural Networks")
    report.append("   - Most sklearn models")
    report.append("")
    report.append("4. Tree-based models (Random Forest, XGBoost):")
    report.append("   - Less sensitive to scaling")
    report.append("   - But scaling still recommended for consistency")
    report.append("")
    
    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return '\n'.join(report)

def main():
    """
    Main function to perform scaling and create report.
    """
    print("=" * 70)
    print("FEATURE SCALING/NORMALIZATION")
    print("=" * 70)
    
    # Load prepared data
    print("\n1. Loading prepared data...")
    df = pd.read_csv('data/integrated_prepared_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    
def process_scaling(df):
    """
    Process scaling for the dataframe.
    Args:
        df: DataFrame to scale
    Returns:
        df_scaled: Scaled DataFrame
        scaling_log: Log of transformations
    """
    # Analyze data
    print("\n2. Analyzing data for scaling...")
    scaling_analysis, numeric_cols = analyze_data_for_scaling(df)
    
    # Apply scaling
    print("\n3. Applying scaling transformations...")
    df_scaled, scaling_log = apply_scaling(df, scaling_analysis, numeric_cols)
    
    return df_scaled, scaling_log

def main():
    """
    Main function to perform scaling and create report.
    """
    print("=" * 70)
    print("FEATURE SCALING/NORMALIZATION")
    print("=" * 70)
    
    # Load prepared data
    print("\n1. Loading prepared data...")
    df = pd.read_csv('data/integrated_prepared_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Process scaling
    df_scaled, scaling_log = process_scaling(df)
    
    # Save scaled data
    print("\n4. Saving scaled data...")
    output_file = 'data/integrated_scaled_data.csv'
    df_scaled.to_csv(output_file, index=False)
    print(f"   OK Saved to: {output_file}")
    
    # Save scaling log
    log_df = pd.DataFrame(scaling_log)
    log_df.to_csv('data/scaling_log.csv', index=False)
    print(f"   OK Scaling log saved to: data/scaling_log.csv")
    
    # Create and save report
    print("\n5. Creating scaling report...")
    report = create_scaling_report(df, df_scaled, scaling_log)
    report_file = 'SCALING_REPORT.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"   OK Report saved to: {report_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SCALING COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Input: integrated_prepared_data.csv")
    print(f"  Output: integrated_scaled_data.csv")
    print(f"  Columns scaled: {len(df_scaled.select_dtypes(include=[np.number]).columns)}")
    print(f"  Methods used: {len(set(e['method'] for e in scaling_log))}")
    print(f"  Rows: {len(df_scaled):,}")
    print(f"  Null values: {df_scaled.isnull().sum().sum()}")
    
    return df_scaled, scaling_log

if __name__ == '__main__':
    df_scaled, scaling_log = main()

