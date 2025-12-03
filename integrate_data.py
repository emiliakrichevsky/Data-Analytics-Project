"""
Data Integration Script

Loads all raw CSV files, fixes MultiIndex issues, and integrates into one unified dataset.
Each row represents a (stock, date) pair with all available raw data columns.
"""

import pandas as pd
import os
from pathlib import Path

DATA_DIR = 'data'
OUTPUT_FILE = 'data/integrated_raw_data.csv'

def load_stock_csv(filepath, stock_name):
    """
    Load stock CSV file with MultiIndex fix.
    
    Args:
        filepath: Path to CSV file
        stock_name: Stock symbol name
    
    Returns:
        DataFrame with Date index and OHLCV columns
    """
    print(f"Loading {stock_name} from {filepath}...")
    
    # Read CSV, skipping MultiIndex header rows (rows 1-2)
    df = pd.read_csv(filepath, skiprows=2)
    
    # Set column names (order: Date, Close, High, Low, Open, Volume)
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    # Convert Date to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    # Remove any rows with missing dates
    df = df.dropna(subset=[df.columns[0]] if len(df.columns) > 0 else [])
    
    print(f"  Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    return df

def load_sp500_csv(filepath):
    """
    Load SP500 CSV file with MultiIndex fix.
    Prefixes columns with 'sp500_' for clarity.
    """
    print(f"Loading SP500 from {filepath}...")
    
    df = pd.read_csv(filepath, skiprows=2)
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df = df.dropna(subset=[df.columns[0]] if len(df.columns) > 0 else [])
    
    # Prefix columns with sp500_
    df.columns = [f'sp500_{col}' for col in df.columns]
    
    print(f"  Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    return df

def load_fred_csv(filepath, column_name):
    """
    Load FRED CSV file (CPI, Fed Funds, etc.).
    These are typically monthly or irregular frequency.
    """
    print(f"Loading {column_name} from {filepath}...")
    
    df = pd.read_csv(filepath)
    
    # FRED CSVs typically have 'date' column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    else:
        # Try first column as date
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0]).sort_index()
    
    # Get the value column (usually the second column or named column)
    if len(df.columns) > 0:
        df = df.iloc[:, [0]]  # Keep first value column
        df.columns = [column_name]
        df = df.dropna()
    
    print(f"  Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    return df

def load_market_csv(filepath, column_names):
    """
    Load market indicator CSV (VIX, Put/Call, etc.).
    
    Args:
        filepath: Path to CSV
        column_names: List of column names to use
    """
    print(f"Loading market data from {filepath}...")
    
    # Check if it's a MultiIndex file (stock format)
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
    
    if 'Ticker' in second_line or 'Price' in first_line:
        # MultiIndex format (like stock files)
        df = pd.read_csv(filepath, skiprows=2)
        # Handle columns - extract just Close price column
        # MultiIndex files have: Date, Close, High, Low, Open, Volume
        if len(df.columns) >= 2:
            # Keep Date (first) and Close (second) columns
            df = df.iloc[:, [0, 1]]  # Keep first two columns
            df.columns = ['Date', column_names[0] if len(column_names) >= 1 else 'value']
        else:
            df.columns = ['Date'] + column_names[:len(df.columns)-1]
    else:
        # Regular CSV
        df = pd.read_csv(filepath)
        if 'Date' not in df.columns and 'date' not in df.columns:
            # Assume first column is date
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
            if len(df.columns) >= len(column_names):
                df.columns = column_names[:len(df.columns)]
            else:
                # If fewer columns than names, just use what we have
                df.columns = column_names[:len(df.columns)]
            return df
    
    # Find date column
    date_col = None
    for col in ['Date', 'date']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    else:
        # Try first column
        if len(df.columns) > 0:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0]).sort_index()
    
    # Keep only specified columns (if we have them)
    available_cols = [col for col in column_names if col in df.columns]
    if available_cols:
        df = df[available_cols]
    elif len(df.columns) > 0:
        # If our column names don't match, rename existing columns
        if len(column_names) >= len(df.columns):
            df.columns = column_names[:len(df.columns)]
    
    print(f"  Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    return df

def load_technical_indicator(filepath, column_name):
    """
    Load technical indicator CSV from Alpha Vantage.
    Format: First row has empty first column and indicator name, dates start from row 2.
    
    Args:
        filepath: Path to CSV file
        column_name: Name for the column
    """
    print(f"Loading technical indicator from {filepath}...")
    
    # Read CSV - first row is header, dates start from row 1 (0-indexed)
    df = pd.read_csv(filepath)
    
    # First column is date (unnamed), second column is the indicator
    if len(df.columns) >= 2:
        # Convert first column to datetime
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        # Rename second column
        df.columns = [column_name]
        df = df.dropna()
        df = df.sort_index()
        # Set index name to Date for easier merging later
        df.index.name = 'Date'
    else:
        print(f"  Warning: Unexpected format, skipping...")
        return None
    
    print(f"  Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    return df

def integrate_all_data():
    """
    Main integration function.
    Loads all raw data sources and creates one unified dataset.
    """
    print("=" * 60)
    print("DATA INTEGRATION - Loading Raw Data Sources")
    print("=" * 60)
    
    # ========== Step 1: Load Stock Data ==========
    print("\n1. Loading Stock Data...")
    stock_data = {}
    stock_symbols = ['GOOGL', 'META']
    
    for symbol in stock_symbols:
        filepath = os.path.join(DATA_DIR, f'{symbol}_raw.csv')
        if os.path.exists(filepath):
            stock_data[symbol] = load_stock_csv(filepath, symbol)
        else:
            print(f"  Warning: {filepath} not found, skipping...")
    
    if not stock_data:
        raise ValueError("No stock data files found!")
    
    # ========== Step 2: Combine Stock Data ==========
    print("\n2. Combining Stock Data...")
    stock_list = []
    for symbol, df in stock_data.items():
        df_copy = df.copy()
        df_copy['stock'] = symbol
        stock_list.append(df_copy.reset_index())
    
    stocks_df = pd.concat(stock_list, ignore_index=True)
    stocks_df = stocks_df.set_index('Date')
    print(f"  Combined stock data: {len(stocks_df)} rows")
    print(f"  Date range: {stocks_df.index.min()} to {stocks_df.index.max()}")
    
    # ========== Step 3: Load SP500 Data ==========
    print("\n3. Loading SP500 Benchmark...")
    sp500_path = os.path.join(DATA_DIR, '^GSPC_raw.csv')
    if os.path.exists(sp500_path):
        sp500_df = load_sp500_csv(sp500_path)
    else:
        print("  Warning: SP500 file not found!")
        sp500_df = None
    
    # ========== Step 4: Load Macro Data (FRED) ==========
    print("\n4. Loading Macro Economic Data...")
    macro_data = {}
    
    fred_files = {
        'CPI_fred.csv': 'CPI',
        'Fed_Funds_Rate_fred.csv': 'Fed_Funds_Rate',
        'GDP_fred.csv': 'GDP',
        'Unemployment_Rate_fred.csv': 'Unemployment_Rate'
    }
    
    for filename, col_name in fred_files.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            macro_data[col_name] = load_fred_csv(filepath, col_name)
        else:
            print(f"  Warning: {filename} not found, skipping...")
    
    # ========== Step 5: Load Market Indicators ==========
    print("\n5. Loading Market Indicators...")
    market_data = {}
    
    # VIX
    vix_path = os.path.join(DATA_DIR, 'vix_term_structure.csv')
    if os.path.exists(vix_path):
        market_data['VIX'] = load_market_csv(vix_path, ['VIX', 'VIX3M'])
    
    # Put/Call Ratio
    pc_path = os.path.join(DATA_DIR, 'put_call_ratio.csv')
    if os.path.exists(pc_path):
        market_data['Put_Call_Ratio'] = load_market_csv(pc_path, ['Put_Call_Ratio'])
    
    # Market Breadth
    mb_path = os.path.join(DATA_DIR, 'market_breadth.csv')
    if os.path.exists(mb_path):
        market_data['Market_Breadth'] = load_market_csv(mb_path, ['Market_Breadth'])
    
    # Sector ETFs
    sectors = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI']
    for sector in sectors:
        sector_path = os.path.join(DATA_DIR, f'sector_{sector}.csv')
        if os.path.exists(sector_path):
            market_data[f'sector_{sector}'] = load_market_csv(sector_path, [f'sector_{sector}'])
    
    # ========== Step 5b: Load Technical Indicators ==========
    print("\n5b. Loading Technical Indicators...")
    technical_data = {}
    
    # SPY indicators
    spy_indicators = {
        'RSI': 'spy_RSI',
        'SMA_50': 'spy_SMA_50',
        'SMA_200': 'spy_SMA_200'
    }
    
    for indicator, col_name in spy_indicators.items():
        tech_path = os.path.join(DATA_DIR, f'technical_SPY_{indicator}.csv')
        if os.path.exists(tech_path):
            technical_data[col_name] = load_technical_indicator(tech_path, col_name)
        else:
            print(f"  Warning: {tech_path} not found, skipping...")
    
    # QQQ indicators
    qqq_indicators = {
        'RSI': 'qqq_RSI',
        'SMA_50': 'qqq_SMA_50',
        'SMA_200': 'qqq_SMA_200'
    }
    
    for indicator, col_name in qqq_indicators.items():
        tech_path = os.path.join(DATA_DIR, f'technical_QQQ_{indicator}.csv')
        if os.path.exists(tech_path):
            technical_data[col_name] = load_technical_indicator(tech_path, col_name)
        else:
            print(f"  Warning: {tech_path} not found, skipping...")
    
    # ========== Step 6: Integrate All Data ==========
    print("\n6. Integrating All Data Sources...")
    
    # Start with stock data (reset index to merge)
    integrated = stocks_df.reset_index()
    
    # Merge SP500 data
    if sp500_df is not None:
        sp500_reset = sp500_df.reset_index()
        integrated = integrated.merge(sp500_reset, on='Date', how='inner')
        print(f"  After SP500 merge: {len(integrated)} rows")
    
    # Merge macro data (forward-fill monthly to daily)
    for col_name, macro_df in macro_data.items():
        macro_reset = macro_df.reset_index()
        # Rename date column if needed
        if 'date' in macro_reset.columns:
            macro_reset = macro_reset.rename(columns={'date': 'Date'})
        
        # Forward-fill macro data (monthly → daily)
        # First, reindex to all trading days in our range
        date_range = pd.date_range(
            start=integrated['Date'].min(),
            end=integrated['Date'].max(),
            freq='D'
        )
        macro_reindexed = macro_df.reindex(date_range).ffill()
        macro_reindexed = macro_reindexed.reset_index()
        macro_reindexed.columns = ['Date', col_name]
        
        # Merge
        integrated = integrated.merge(macro_reindexed, on='Date', how='left')
        # Forward-fill any remaining NaN
        integrated[col_name] = integrated[col_name].ffill()
        print(f"  After {col_name} merge: {len(integrated)} rows")
    
    # Merge market indicators (already daily, direct merge)
    for col_name, market_df in market_data.items():
        market_reset = market_df.reset_index()
        # Rename date column if needed
        date_col = None
        for dc in ['Date', 'date']:
            if dc in market_reset.columns:
                date_col = dc
                break
        
        if date_col and date_col != 'Date':
            market_reset = market_reset.rename(columns={date_col: 'Date'})
        
        if 'Date' in market_reset.columns:
            integrated = integrated.merge(market_reset, on='Date', how='left')
            print(f"  After {col_name} merge: {len(integrated)} rows")
    
    # Merge technical indicators (already daily, direct merge)
    for col_name, tech_df in technical_data.items():
        if tech_df is not None:
            tech_reset = tech_df.reset_index()
            # Rename date column if needed
            date_col = None
            for dc in ['Date', 'date']:
                if dc in tech_reset.columns:
                    date_col = dc
                    break
            
            if date_col and date_col != 'Date':
                tech_reset = tech_reset.rename(columns={date_col: 'Date'})
            
            if 'Date' in tech_reset.columns:
                integrated = integrated.merge(tech_reset, on='Date', how='left')
                print(f"  After {col_name} merge: {len(integrated)} rows")
    
    # ========== Step 7: Final Cleanup ==========
    print("\n7. Final Cleanup...")
    
    # Sort by Date, then stock
    integrated = integrated.sort_values(['Date', 'stock']).reset_index(drop=True)
    
    # Set Date as index (optional, but useful)
    # integrated = integrated.set_index('Date')
    
    print(f"\nFinal integrated dataset:")
    print(f"  Total rows: {len(integrated)}")
    print(f"  Columns: {len(integrated.columns)}")
    print(f"  Date range: {integrated['Date'].min()} to {integrated['Date'].max()}")
    print(f"  Stocks: {integrated['stock'].unique().tolist()}")
    print(f"\nColumns: {integrated.columns.tolist()}")
    
    # ========== Step 8: Save ==========
    print(f"\n8. Saving to {OUTPUT_FILE}...")
    os.makedirs(DATA_DIR, exist_ok=True)
    integrated.to_csv(OUTPUT_FILE, index=False)
    print("  Saved successfully!")
    
    return integrated

if __name__ == '__main__':
    integrated_df = integrate_all_data()
    
    print("\n" + "=" * 60)
    print("INTEGRATION COMPLETE")
    print("=" * 60)
    print(f"\nDataset saved to: {OUTPUT_FILE}")
    print(f"\nFirst few rows:")
    print(integrated_df.head())
    print(f"\nData types:")
    print(integrated_df.dtypes)

