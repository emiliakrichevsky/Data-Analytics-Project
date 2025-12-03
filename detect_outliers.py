"""
Outlier Detection Script

Detects outliers in the integrated prepared dataset using multiple methods.
Provides concrete examples of detected outliers.
"""

import pandas as pd
import numpy as np
from typing import Tuple

def detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Detect outliers using Z-score method.
    Values beyond threshold standard deviations are outliers.
    z-scores should be a Series of absolute z-scores for the column.
    outliers is a boolean Series where True indicates an outlier.
    """
    # Coerce values to numeric (safest for mixed-type columns)
    vals = pd.to_numeric(df[col], errors='coerce')
    mean = vals.mean()
    std = vals.std()

    # If std is zero or NaN (constant or all-NaN column), return no outliers
    if std == 0 or np.isnan(std):
        outliers = pd.Series(False, index=df.index)
        z_scores = pd.Series(0.0, index=df.index)
        return outliers, z_scores

    # compute absolute z-scores
    z_scores = (vals - mean).abs() / std
    outliers = z_scores > threshold

    return outliers, z_scores

def detect_outliers_iqr(df: pd.DataFrame, col: str, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR (Interquartile Range) method.
    Values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR are outliers.
    """
    # Coerce to numeric to avoid errors with mixed-type columns
    vals = pd.to_numeric(df[col], errors='coerce')
    Q1 = vals.quantile(0.25)
    Q3 = vals.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = (vals < lower_bound) | (vals > upper_bound)

    # ensure same index as original df
    outliers = pd.Series(outliers, index=df.index)

    return outliers

# def main():
#     """
#     Main outlier detection function.
#     """
#     print("=" * 70)
#     print("OUTLIER DETECTION ANALYSIS")
#     print("=" * 70)
    
#     # Load data
#     print("\n1. Loading data...")
#     df = pd.read_csv('data/integrated_prepared_data.csv')
#     df['Date'] = pd.to_datetime(df['Date'])
#     print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    
def remove_outliers(df, zscore_results=None, iqr_results=None, method='zscore'):
    """
    Remove outliers from the dataframe.
    Args:
        df: DataFrame
        zscore_results: Results from Z-score detection
        iqr_results: Results from IQR detection
        method: 'zscore' or 'iqr'
    Returns:
        df_clean: DataFrame with outliers removed
    """
    df_clean = df.copy()
    outlier_indices = set()
    
    if method == 'zscore' and zscore_results:
        for col, stats in zscore_results.items():
            outlier_indices.update(stats['outlier_indices'])
    elif method == 'iqr' and iqr_results:
        for col, stats in iqr_results.items():
            outlier_indices.update(stats['outlier_indices'])
            
    if outlier_indices:
        df_clean = df_clean.drop(index=list(outlier_indices))
        
    return df_clean

def analyze_outliers(df):
    """
    Analyze outliers in the dataframe.
    Args:
        df: DataFrame
    Returns:
        zscore_results, iqr_results
    """
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Date']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"\n2. Analyzing {len(numeric_cols)} numeric columns for outliers...")
    
    # ========== DETECTION BY METHOD ==========
    print("\n" + "=" * 70)
    print("OUTLIER DETECTION - Z-SCORE METHOD (Threshold: 3 std)")
    print("=" * 70)
    
    zscore_results = {}
    for col in numeric_cols:
        outliers, z_scores = detect_outliers_zscore(df, col, threshold=3)
        outlier_count = outliers.sum()
        if outlier_count > 0:
            zscore_results[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100,
                'max_zscore': z_scores.max(),
                'outlier_indices': df[outliers].index.tolist()
            }
    
    print(f"\nColumns with outliers (Z-score > 3):")
    if zscore_results:
        print(f"{'Column':<25} {'Outliers':<12} {'Percentage':<12} {'Max Z-Score'}")
        print("-" * 70)
        for col, stats in sorted(zscore_results.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"{col:<25} {stats['count']:<12} {stats['percentage']:.2f}%      {stats['max_zscore']:.2f}")
    else:
        print("   No outliers found with Z-score method (threshold: 3)")
    
    print("\n" + "=" * 70)
    print("OUTLIER DETECTION - IQR METHOD (1.5 * IQR)")
    print("=" * 70)
    
    iqr_results = {}
    for col in numeric_cols:
        outliers = detect_outliers_iqr(df, col, multiplier=1.5)
        outlier_count = outliers.sum()
        if outlier_count > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_results[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_indices': df[outliers].index.tolist()
            }
    
    print(f"\nColumns with outliers (IQR method):")
    if iqr_results:
        print(f"{'Column':<25} {'Outliers':<12} {'Percentage':<12} {'Range'}")
        print("-" * 70)
        for col, stats in sorted(iqr_results.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            print(f"{col:<25} {stats['count']:<12} {stats['percentage']:.2f}%      [{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]")
    else:
        print("   No outliers found with IQR method")
        
    return zscore_results, iqr_results

def main():
    """
    Main outlier detection function.
    """
    print("=" * 70)
    print("OUTLIER DETECTION ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv('data/integrated_prepared_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    # Determine numeric columns here as well for use later in main
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Date']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    zscore_results, iqr_results = analyze_outliers(df)
    
    # ========== CONCRETE EXAMPLES ==========
    print("\n" + "=" * 70)
    print("CONCRETE OUTLIER EXAMPLES")
    print("=" * 70)
    
    # Example 1: Volume outliers (likely to have outliers)
    if 'Volume' in zscore_results:
        print("\nEXAMPLE 1: Volume Outliers (Extreme Trading Days)")
        print("-" * 70)
        
        vol_outliers = df.loc[zscore_results['Volume']['outlier_indices'][:5]]
        vol_outliers = vol_outliers.sort_values('Volume', ascending=False)
        
        print(f"\nTop 5 Volume Outliers:")
        print(f"{'Date':<12} {'Stock':<6} {'Volume':<20} {'Z-Score':<10} {'Context'}")
        print("-" * 70)
        
        for idx, row in vol_outliers.iterrows():
            date = row['Date'].strftime('%Y-%m-%d')
            stock = row['stock']
            volume = row['Volume']
            zscore = abs((volume - df['Volume'].mean()) / df['Volume'].std())
            close = row['Close']
            
            # Context: Is it a major move?
            print(f"{date:<12} {stock:<6} {volume:>18,} {zscore:>8.2f} Close=${close:.2f}")
    
    # Example 2: Close price outliers
    if 'Close' in zscore_results:
        print("\nEXAMPLE 2: Close Price Outliers (Extreme Stock Prices)")
        print("-" * 70)
        
        close_outliers = df.loc[zscore_results['Close']['outlier_indices'][:5]]
        close_outliers = close_outliers.sort_values('Close', ascending=False)
        
        print(f"\nTop 5 Close Price Outliers (Highest):")
        print(f"{'Date':<12} {'Stock':<6} {'Close ($)':<12} {'Z-Score':<10} {'Volume':<15}")
        print("-" * 70)
        
        for idx, row in close_outliers.iterrows():
            date = row['Date'].strftime('%Y-%m-%d')
            stock = row['stock']
            close = row['Close']
            zscore = abs((close - df['Close'].mean()) / df['Close'].std())
            volume = row['Volume']
            print(f"{date:<12} {stock:<6} ${close:>9.2f} {zscore:>8.2f} {volume:>14,}")
        
        # Also show lowest
        close_outliers_low = df.loc[zscore_results['Close']['outlier_indices']]
        close_outliers_low = close_outliers_low.sort_values('Close', ascending=True)
        
        print(f"\nLowest Close Price Outliers:")
        print(f"{'Date':<12} {'Stock':<6} {'Close ($)':<12} {'Z-Score':<10} {'Context'}")
        print("-" * 70)
        
        for idx, row in close_outliers_low.head(3).iterrows():
            date = row['Date'].strftime('%Y-%m-%d')
            stock = row['stock']
            close = row['Close']
            zscore = abs((close - df['Close'].mean()) / df['Close'].std())
            print(f"{date:<12} {stock:<6} ${close:>9.2f} {zscore:>8.2f} (IPO period)")
    
    # Example 3: VIX outliers (market stress)
    if 'VIX' in zscore_results:
        print("\nEXAMPLE 3: VIX Outliers (Market Stress/Fear)")
        print("-" * 70)
        
        vix_outliers = df.loc[zscore_results['VIX']['outlier_indices'][:5]]
        vix_outliers = vix_outliers.sort_values('VIX', ascending=False)
        
        print(f"\nTop 5 VIX Outliers (High Fear/Volatility):")
        print(f"{'Date':<12} {'VIX':<10} {'Z-Score':<10} {'Stock':<6} {'Close':<10} {'Context'}")
        print("-" * 70)
        
        for idx, row in vix_outliers.iterrows():
            date = row['Date'].strftime('%Y-%m-%d')
            vix = row['VIX']
            zscore = abs((vix - df['VIX'].mean()) / df['VIX'].std())
            stock = row['stock']
            close = row['Close']
            
            # Check if this is a known market event
            context = ""
            if pd.to_datetime(date).year == 2008 or pd.to_datetime(date).year == 2009:
                context = "Financial Crisis"
            elif pd.to_datetime(date).year == 2020:
                context = "COVID-19"
            elif pd.to_datetime(date).year == 2022:
                context = "Market Volatility"
            
            print(f"{date:<12} {vix:>8.2f} {zscore:>8.2f} {stock:<6} ${close:>7.2f} {context}")
    
    # Example 4: Row with multiple outliers
    print("\nEXAMPLE 4: Rows with Multiple Outliers")
    print("-" * 70)
    
    # Count outliers per row
    all_outlier_indices = set()
    for col, stats in zscore_results.items():
        all_outlier_indices.update(stats['outlier_indices'])
    
    row_outlier_counts = {}
    for idx in all_outlier_indices:
        count = sum(1 for col in zscore_results.keys() 
                   if idx in zscore_results[col]['outlier_indices'])
        row_outlier_counts[idx] = count
    
    # Find rows with most outliers
    if row_outlier_counts:
        top_outlier_rows = sorted(row_outlier_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"\nRows with Most Outlier Columns:")
        print(f"{'Date':<12} {'Stock':<6} {'Outlier Count':<15} {'Example Outliers'}")
        print("-" * 70)
        
        for idx, count in top_outlier_rows:
            row = df.iloc[idx]
            date = row['Date'].strftime('%Y-%m-%d')
            stock = row['stock']
            
            # Find which columns are outliers
            outlier_cols = [col for col in zscore_results.keys() 
                          if idx in zscore_results[col]['outlier_indices']]
            
            print(f"{date:<12} {stock:<6} {count:<15} {', '.join(outlier_cols[:3])}")
    
    # ========== COMPARISON: Z-SCORE vs IQR ==========
    print("\n" + "=" * 70)
    print("METHOD COMPARISON: Z-SCORE vs IQR")
    print("=" * 70)
    
    if zscore_results and iqr_results:
        common_cols = set(zscore_results.keys()) & set(iqr_results.keys())
        
        print(f"\nColumns detected by both methods: {len(common_cols)}")
        print(f"\nSample comparison for Volume:")
        
        if 'Volume' in common_cols:
            z_count = zscore_results['Volume']['count']
            iqr_count = iqr_results['Volume']['count']
            
            print(f"  Z-Score method: {z_count} outliers")
            print(f"  IQR method: {iqr_count} outliers")
            print(f"  Difference: {abs(z_count - iqr_count)} outliers")
            print(f"\n  Why difference?")
            print(f"    - Z-Score: Based on mean/std (sensitive to extreme values)")
            print(f"    - IQR: Based on quartiles (more robust to extremes)")
    
    # ========== OUTLIER STATISTICS ==========
    print("\n" + "=" * 70)
    print("OUTLIER STATISTICS SUMMARY")
    print("=" * 70)
    
    total_outliers_zscore = sum(stats['count'] for stats in zscore_results.values())
    total_outliers_iqr = sum(stats['count'] for stats in iqr_results.values())
    
    print(f"\nZ-Score Method:")
    print(f"  Columns with outliers: {len(zscore_results)}")
    print(f"  Total outlier instances: {total_outliers_zscore:,}")
    print(f"  Percentage of dataset: {(total_outliers_zscore / (len(df) * len(numeric_cols))) * 100:.2f}%")
    
    print(f"\nIQR Method:")
    print(f"  Columns with outliers: {len(iqr_results)}")
    print(f"  Total outlier instances: {total_outliers_iqr:,}")
    print(f"  Percentage of dataset: {(total_outliers_iqr / (len(df) * len(numeric_cols))) * 100:.2f}%")
    
    # ========== SAVE RESULTS ==========
    print("\n" + "=" * 70)
    print("SAVING OUTLIER DETECTION RESULTS")
    print("=" * 70)
    
    # Create summary DataFrame
    outlier_summary = []
    for col in numeric_cols:
        z_outliers = zscore_results.get(col, {}).get('count', 0)
        iqr_outliers = iqr_results.get(col, {}).get('count', 0)
        z_pct = zscore_results.get(col, {}).get('percentage', 0)
        iqr_pct = iqr_results.get(col, {}).get('percentage', 0)
        
        outlier_summary.append({
            'column': col,
            'zscore_outliers': z_outliers,
            'zscore_percentage': z_pct,
            'iqr_outliers': iqr_outliers,
            'iqr_percentage': iqr_pct
        })
    
    summary_df = pd.DataFrame(outlier_summary)
    summary_df = summary_df.sort_values('zscore_outliers', ascending=False)
    summary_df.to_csv('data/outlier_detection_summary.csv', index=False)
    print(f"\n   OK Outlier summary saved to: data/outlier_detection_summary.csv")
    
    return zscore_results, iqr_results, df

if __name__ == '__main__':
    zscore_results, iqr_results, df = main()
    
    print("\n" + "=" * 70)
    print("OUTLIER DETECTION COMPLETE")
    print("=" * 70)

