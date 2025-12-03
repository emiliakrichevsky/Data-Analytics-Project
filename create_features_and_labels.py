"""
Feature Engineering and Label Construction

Creates engineered features (X) and target variable (y) from integrated prepared data.
Saves to a separate CSV file: ml_features_and_labels.csv
"""

import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Engineer features from raw data.
    Returns DataFrame with new feature columns.
    """
    print("\n2. Engineering features...")
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['stock', 'Date']).reset_index(drop=True)
    
    features_log = []
    
    # ========== STOCK FEATURES (Grouped by stock) ==========
    print("\n   a) Stock price features...")
    
    # Daily return
    df['daily_return'] = df.groupby('stock')['Close'].pct_change()
    features_log.append('daily_return: Stock daily return (pct_change)')
    
    # Rolling returns (1 week = 5 days, 1 month = 21 days, 3 months = 63 days)
    print("      - Rolling returns...")
    df['r_1W'] = df.groupby('stock')['Close'].pct_change(periods=5)
    df['r_1M'] = df.groupby('stock')['Close'].pct_change(periods=21)
    df['r_3M'] = df.groupby('stock')['Close'].pct_change(periods=63)
    features_log.append('r_1W: 1-week (5-day) return')
    features_log.append('r_1M: 1-month (21-day) return')
    features_log.append('r_3M: 3-month (63-day) return')
    
    # Volatility (rolling std of daily returns)
    print("      - Volatility...")
    df['vol_1M'] = df.groupby('stock')['daily_return'].rolling(window=21, min_periods=1).std().reset_index(0, drop=True)
    features_log.append('vol_1M: 1-month volatility (rolling std of daily returns)')
    
    # Moving Average Ratios
    print("      - Moving average ratios...")
    df['MA20'] = df.groupby('stock')['Close'].rolling(window=20, min_periods=1).mean().reset_index(0, drop=True)
    df['MA50'] = df.groupby('stock')['Close'].rolling(window=50, min_periods=1).mean().reset_index(0, drop=True)
    df['MA20_ratio'] = df['Close'] / df['MA20']
    df['MA50_ratio'] = df['Close'] / df['MA50']
    features_log.append('MA20_ratio: Close / MA20')
    features_log.append('MA50_ratio: Close / MA50')
    
    # Price range (High - Low) / Close
    print("      - Price range...")
    df['HL_range'] = (df['High'] - df['Low']) / df['Close']
    features_log.append('HL_range: (High - Low) / Close')
    
    # Volume z-score (rolling)
    print("      - Volume z-score...")
    vol_mean_20 = df.groupby('stock')['Volume'].rolling(window=20, min_periods=1).mean().reset_index(0, drop=True)
    vol_std_20 = df.groupby('stock')['Volume'].rolling(window=20, min_periods=1).std().reset_index(0, drop=True)
    df['vol_z'] = (df['Volume'] - vol_mean_20) / (vol_std_20 + 1e-8)  # Add small epsilon to avoid division by zero
    features_log.append('vol_z: Volume z-score (Volume - mean(20)) / std(20)')
    
    # ========== SP500 FEATURES ==========
    print("\n   b) SP500 features...")
    
    # SP500 data is same for all stocks on same date, so compute once per date
    sp500_by_date = df.groupby('Date')['sp500_Close'].first().reset_index()
    sp500_by_date = sp500_by_date.sort_values('Date')
    
    # Compute SP500 daily return
    sp500_by_date['sp500_daily_return'] = sp500_by_date['sp500_Close'].pct_change()
    
    # Map to main dataframe
    sp500_return_map = dict(zip(sp500_by_date['Date'], sp500_by_date['sp500_daily_return']))
    df['sp500_daily_return'] = df['Date'].map(sp500_return_map)
    
    # Compute SP500 volatility (rolling std of returns)
    sp500_by_date['sp500_vol_1M'] = sp500_by_date['sp500_daily_return'].rolling(window=21, min_periods=1).std()
    sp500_vol_map = dict(zip(sp500_by_date['Date'], sp500_by_date['sp500_vol_1M']))
    df['sp500_vol_1M'] = df['Date'].map(sp500_vol_map)
    
    features_log.append('sp500_daily_return: SP500 daily return')
    features_log.append('sp500_vol_1M: SP500 1-month volatility')
    
    # ========== RELATIVE FEATURES ==========
    print("\n   c) Relative features (stock vs SP500)...")
    
    df['relative_return'] = df['daily_return'] - df['sp500_daily_return']
    df['volatility_ratio'] = df['vol_1M'] / (df['sp500_vol_1M'] + 1e-8)
    features_log.append('relative_return: stock_return - sp500_return')
    features_log.append('volatility_ratio: stock_vol / sp500_vol')
    
    # ========== MACRO FEATURES ==========
    print("\n   d) Macro features...")
    
    # CPI change (month-over-month) - same for all stocks on same date
    macro_by_date = df.groupby('Date').agg({
        'CPI': 'first',
        'Fed_Funds_Rate': 'first'
    }).reset_index()
    macro_by_date = macro_by_date.sort_values('Date')
    
    macro_by_date['CPI_chg'] = macro_by_date['CPI'].pct_change()
    macro_by_date['FedFunds_chg'] = macro_by_date['Fed_Funds_Rate'].diff()
    
    cpi_chg_map = dict(zip(macro_by_date['Date'], macro_by_date['CPI_chg']))
    fed_chg_map = dict(zip(macro_by_date['Date'], macro_by_date['FedFunds_chg']))
    
    df['CPI_chg'] = df['Date'].map(cpi_chg_map)
    df['FedFunds_chg'] = df['Date'].map(fed_chg_map)
    
    features_log.append('CPI_chg: CPI percentage change')
    features_log.append('FedFunds_chg: Federal Funds Rate change (absolute)')
    
    # Direct macro indicators (already in data)
    df['VIX_t'] = df['VIX']
    df['FedFunds_t'] = df['Fed_Funds_Rate']
    features_log.append('VIX_t: VIX index (direct use)')
    features_log.append('FedFunds_t: Federal Funds Rate (direct use)')
    
    # ========== MARKET FEATURES ==========
    print("\n   e) Market features...")
    
    # These are already in the data, just rename for clarity
    df['Put_Call_Ratio_t'] = df['Put_Call_Ratio']
    df['Market_Breadth_t'] = df['Market_Breadth']
    features_log.append('Put_Call_Ratio_t: Put/Call Ratio (direct use)')
    features_log.append('Market_Breadth_t: Market Breadth (direct use)')
    
    # ========== TECHNICAL INDICATORS (already available) ==========
    print("\n   f) Technical indicators...")
    
    # SPY/QQQ indicators are already in the data
    df['spy_RSI_t'] = df['spy_RSI']
    df['spy_SMA_50_t'] = df['spy_SMA_50']
    df['spy_SMA_200_t'] = df['spy_SMA_200']
    df['qqq_RSI_t'] = df['qqq_RSI']
    df['qqq_SMA_50_t'] = df['qqq_SMA_50']
    df['qqq_SMA_200_t'] = df['qqq_SMA_200']
    features_log.append('spy_RSI_t: SPY RSI (direct use)')
    features_log.append('spy_SMA_50_t, spy_SMA_200_t: SPY moving averages')
    features_log.append('qqq_RSI_t: QQQ RSI (direct use)')
    features_log.append('qqq_SMA_50_t, qqq_SMA_200_t: QQQ moving averages')
    
    # ========== SECTOR FEATURES ==========
    print("\n   f) Sector features...")
    
    # Sector ETFs are already in the data
    df['sector_XLK_t'] = df['sector_XLK']
    df['sector_XLF_t'] = df['sector_XLF']
    df['sector_XLV_t'] = df['sector_XLV']
    df['sector_XLE_t'] = df['sector_XLE']
    df['sector_XLI_t'] = df['sector_XLI']
    features_log.append('sector_XLK_t, sector_XLF_t, etc.: Sector ETF prices (direct use)')
    
    print(f"\n   Total features engineered: {len(features_log)}")
    
    return df, features_log

def construct_labels(df):
    """
    Construct target variable y.
    y = 1 if stock forward return (21 days) > SP500 forward return (21 days), else 0
    """
    print("\n3. Constructing labels (target variable y)...")
    
    df = df.copy()
    df = df.sort_values(['stock', 'Date']).reset_index(drop=True)
    
    # Forward 21-day returns for stock
    print("   - Computing forward 21-day returns for stocks...")
    df['stock_fwd_ret_21d'] = df.groupby('stock')['Close'].apply(
        lambda x: x.shift(-21) / x - 1
    ).reset_index(0, drop=True)
    
    # Forward 21-day returns for SP500 (same for all stocks on same date)
    print("   - Computing forward 21-day returns for SP500...")
    sp500_fwd = df.groupby('Date')['sp500_Close'].first().shift(-21) / df.groupby('Date')['sp500_Close'].first() - 1
    df['sp500_fwd_ret_21d'] = df['Date'].map(sp500_fwd)
    
    # Binary label: 1 if stock outperforms SP500, else 0
    print("   - Creating binary label...")
    df['y'] = (df['stock_fwd_ret_21d'] > df['sp500_fwd_ret_21d']).astype(int)
    
    # Handle NaN values (last 21 days don't have forward returns)
    nan_count = df['y'].isna().sum()
    print(f"   - Rows without labels (last 21 days): {nan_count}")
    print(f"   - Rows with labels: {len(df) - nan_count}")
    
    return df

def main():
    """
    Main function to engineer features and construct labels.
    """
    print("=" * 70)
    print("FEATURE ENGINEERING AND LABEL CONSTRUCTION")
    print("=" * 70)
    
    # Load prepared data (unscaled)
    print("\n1. Loading prepared data...")
    df = pd.read_csv('data/integrated_prepared_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"   Stocks: {df['stock'].unique()}")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
def process_data(df):
    """
    Process the dataframe to engineer features and construct labels.
    Args:
        df: DataFrame with prepared data
    Returns:
        df_ml: DataFrame with features and labels
        features_log: List of created features
    """
    # Engineer features
    df_features, features_log = engineer_features(df)
    
    # Construct labels
    df_final = construct_labels(df_features)
    
    # Select feature columns and label
    print("\n4. Selecting feature columns...")
    
    # Feature columns (all engineered features)
    feature_cols = [
        'Date', 'stock',  # Identifiers
        # Stock features
        'daily_return', 'r_1W', 'r_1M', 'r_3M', 'vol_1M',
        'MA20_ratio', 'MA50_ratio', 'HL_range', 'vol_z',
        # SP500 features
        'sp500_daily_return', 'sp500_vol_1M',
        # Relative features
        'relative_return', 'volatility_ratio',
        # Macro features
        'VIX_t', 'FedFunds_t', 'CPI_chg', 'FedFunds_chg',
        # Market features
        'Put_Call_Ratio_t', 'Market_Breadth_t',
        # Technical indicators
        'spy_RSI_t', 'spy_SMA_50_t', 'spy_SMA_200_t',
        'qqq_RSI_t', 'qqq_SMA_50_t', 'qqq_SMA_200_t',
        # Sector features
        'sector_XLK_t', 'sector_XLF_t', 'sector_XLV_t', 'sector_XLE_t', 'sector_XLI_t',
        # Target variable
        'y', 'stock_fwd_ret_21d', 'sp500_fwd_ret_21d'  # Keep forward returns for reference
    ]
    
    # Check which columns exist
    available_cols = [col for col in feature_cols if col in df_final.columns]
    missing_cols = [col for col in feature_cols if col not in df_final.columns]
    
    if missing_cols:
        print(f"   Warning: Missing columns: {missing_cols}")
    
    df_ml = df_final[available_cols].copy()
    
    print(f"\n   Selected {len(available_cols)} columns")
    print(f"   Features: {len(available_cols) - 4} (excluding Date, stock, and target variables)")
    
    return df_ml, features_log

def main():
    """
    Main function to engineer features and construct labels.
    """
    print("=" * 70)
    print("FEATURE ENGINEERING AND LABEL CONSTRUCTION")
    print("=" * 70)
    
    # Load prepared data (unscaled)
    print("\n1. Loading prepared data...")
    df = pd.read_csv('data/integrated_prepared_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"   Stocks: {df['stock'].unique()}")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Process data
    df_ml, features_log = process_data(df)
    
    # Save to CSV
    print("\n5. Saving ML dataset...")
    output_file = 'data/ml_features_and_labels.csv'
    df_ml.to_csv(output_file, index=False, float_format='%.6f')
    print(f"   OK Saved to: {output_file}")
    print(f"   Rows: {len(df_ml):,}")
    print(f"   Columns: {len(df_ml.columns)}")
    print(f"   Null values: {df_ml.isnull().sum().sum()}")
    
    # Save feature log
    print("\n6. Saving feature engineering log...")
    with open('data/feature_engineering_log.txt', 'w', encoding='utf-8') as f:
        f.write("FEATURE ENGINEERING LOG\n")
        f.write("=" * 70 + "\n\n")
        for i, feat in enumerate(features_log, 1):
            f.write(f"{i}. {feat}\n")
    print(f"   OK Saved to: data/feature_engineering_log.txt")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal features created: {len(features_log)}")
    print(f"\nLabel distribution:")
    if 'y' in df_ml.columns:
        label_counts = df_ml['y'].value_counts(dropna=True)
        print(f"  y=0 (underperform): {label_counts.get(0, 0):,}")
        print(f"  y=1 (outperform): {label_counts.get(1, 0):,}")
        print(f"  y=NaN (last 21 days): {df_ml['y'].isna().sum():,}")
        print(f"  Outperform rate: {(label_counts.get(1, 0) / (label_counts.get(0, 0) + label_counts.get(1, 0)) * 100) if (label_counts.get(0, 0) + label_counts.get(1, 0)) > 0 else 0:.2f}%")
    
    print(f"\nFeature columns:")
    available_cols = df_ml.columns.tolist()
    for i, col in enumerate(available_cols[:10], 1):
        print(f"  {i}. {col}")
    if len(available_cols) > 10:
        print(f"  ... and {len(available_cols) - 10} more")
    
    return df_ml

if __name__ == '__main__':
    df_ml = main()
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

