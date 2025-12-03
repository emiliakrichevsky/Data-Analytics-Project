"""
Data Preparation Script

Handles null values in integrated dataset using column-specific methods.
Each column with nulls is analyzed and handled appropriately based on its
data characteristics (temporal, macro, statistical).
"""

import pandas as pd
import numpy as np

def prepare_data():
    """
    Prepare integrated data by handling null values with appropriate methods.
    """
    print("=" * 70)
    print("DATA PREPARATION - NULL VALUE HANDLING")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading integrated_raw_data.csv...")
    df = pd.read_csv('data/integrated_raw_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Sort by stock and date for proper time series handling
    df = df.sort_values(['stock', 'Date']).reset_index(drop=True)
    
    # ========== IDENTIFY NULL COLUMNS ==========
    print("\n2. Identifying columns with null values...")
    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0].sort_values(ascending=False)
    
    if len(null_columns) == 0:
        print("   ✓ No null values found!")
        return df
    
    print(f"   Found {len(null_columns)} columns with null values:")
    for col, count in null_columns.items():
        pct = (count / len(df)) * 100
        print(f"     - {col}: {count:,} nulls ({pct:.2f}%)")
    
    # ========== HANDLE EACH COLUMN ==========
    print("\n3. Handling null values column by column...")
    print("-" * 70)
    
    handling_log = []
    
    for col in null_columns.index:
        null_count_before = df[col].isnull().sum()
        print(f"\n   Column: {col}")
        print(f"   Null count before: {null_count_before}")
        
        # Determine handling method based on column characteristics
        method = determine_handling_method(col, df)
        print(f"   Method: {method['name']}")
        print(f"   Reason: {method['reason']}")
        
        # Apply handling method
        df = apply_handling_method(df, col, method)
        
        null_count_after = df[col].isnull().sum()
        handled_count = null_count_before - null_count_after
        
        print(f"   Null count after: {null_count_after}")
        print(f"   Handled: {handled_count} nulls")
        
        handling_log.append({
            'column': col,
            'nulls_before': null_count_before,
            'method': method['name'],
            'nulls_after': null_count_after,
            'handled': handled_count
        })
    
    # ========== FINAL NULL CHECK ==========
    print("\n4. Final null value check...")
    print("-" * 70)
    final_nulls = df.isnull().sum().sum()
    if final_nulls == 0:
        print("   ✓ All null values have been handled!")
    else:
        remaining = df.isnull().sum()
        remaining = remaining[remaining > 0]
        print(f"   ⚠ {final_nulls} null values remain in {len(remaining)} columns:")
        for col, count in remaining.items():
            print(f"     - {col}: {count} nulls")
    
    # ========== SAVE PREPARED DATA ==========
    print("\n5. Saving prepared data...")
    output_file = 'data/integrated_prepared_data.csv'
    df.to_csv(output_file, index=False)
    print(f"   ✓ Saved to: {output_file}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    
    # ========== HANDLING SUMMARY ==========
    print("\n6. Handling Summary...")
    print("-" * 70)
    summary_df = pd.DataFrame(handling_log)
    print("\n" + summary_df.to_string(index=False))
    
    # Save handling log
    summary_df.to_csv('data/null_handling_log.csv', index=False)
    print("\n   ✓ Handling log saved to: data/null_handling_log.csv")
    
    return df

def determine_handling_method(col, df):
    """
    Determine the appropriate null handling method for a column.
    
    Returns:
        dict with 'name', 'reason', and method parameters
    """
    col_data = df[col]
    null_indices = col_data[col_data.isnull()].index
    
    # Check column characteristics
    is_numeric = pd.api.types.is_numeric_dtype(col_data)
    is_date_indexed = 'Date' in df.columns
    
    # CPI - Monthly Consumer Price Index
    if col == 'CPI':
        # Check if nulls are at start (need backward-fill) or scattered
        if null_indices.min() < 100:  # Nulls at start
            return {
                'name': 'Backward Fill (Grouped by Stock)',
                'reason': 'Monthly macro data - nulls at start, backward-fill from next available value within each stock',
                'group_by_stock': True,
                'direction': 'backward'
            }
        else:
            return {
                'name': 'Forward Fill then Backward Fill',
                'reason': 'Monthly macro data - fill from both directions',
                'group_by_stock': True,
                'direction': 'both'
            }
    
    # GDP - Quarterly Gross Domestic Product
    elif col == 'GDP':
        # GDP is quarterly, nulls likely at start
        if null_indices.min() < 100:  # Nulls at start
            return {
                'name': 'Backward Fill (Grouped by Stock)',
                'reason': 'Quarterly macro data - nulls at start, backward-fill from next available value within each stock',
                'group_by_stock': True,
                'direction': 'backward'
            }
        else:
            return {
                'name': 'Forward Fill then Backward Fill',
                'reason': 'Quarterly macro data - fill from both directions',
                'group_by_stock': True,
                'direction': 'both'
            }
    
    # Unemployment_Rate - Monthly unemployment data
    elif col == 'Unemployment_Rate':
        if null_indices.min() < 100:  # Nulls at start
            return {
                'name': 'Backward Fill (Grouped by Stock)',
                'reason': 'Monthly macro data - nulls at start, backward-fill from next available value within each stock',
                'group_by_stock': True,
                'direction': 'backward'
            }
        else:
            return {
                'name': 'Forward Fill then Backward Fill',
                'reason': 'Monthly macro data - fill from both directions',
                'group_by_stock': True,
                'direction': 'both'
            }
    
    # Default for numeric columns
    elif is_numeric:
        # Check null pattern
        if null_indices.min() < len(df) * 0.1:  # Nulls at start (< 10%)
            return {
                'name': 'Forward Fill',
                'reason': 'Numeric column with nulls at start - forward-fill',
                'group_by_stock': True,
                'direction': 'forward'
            }
        elif null_indices.max() > len(df) * 0.9:  # Nulls at end (> 90%)
            return {
                'name': 'Backward Fill',
                'reason': 'Numeric column with nulls at end - backward-fill',
                'group_by_stock': True,
                'direction': 'backward'
            }
        else:
            # Scattered nulls - use interpolation or mean
            return {
                'name': 'Interpolation (Linear)',
                'reason': 'Numeric column with scattered nulls - linear interpolation',
                'group_by_stock': True,
                'interpolation': 'linear'
            }
    
    # Default for non-numeric
    else:
        return {
            'name': 'Forward Fill',
            'reason': 'Non-numeric column - forward-fill default',
            'group_by_stock': True,
            'direction': 'forward'
        }

def apply_handling_method(df, col, method):
    """
    Apply the specified handling method to a column.
    """
    method_name = method['name']
    
    if 'Forward Fill' in method_name:
        if method.get('group_by_stock', False):
            df[col] = df.groupby('stock')[col].ffill()
        else:
            df[col] = df[col].ffill()
    
    elif 'Backward Fill' in method_name:
        if method.get('group_by_stock', False):
            df[col] = df.groupby('stock')[col].bfill()
        else:
            df[col] = df[col].bfill()
    
    elif 'Forward Fill then Backward Fill' in method_name:
        if method.get('group_by_stock', False):
            df[col] = df.groupby('stock')[col].ffill().bfill()
        else:
            df[col] = df[col].ffill().bfill()
    
    elif 'Interpolation' in method_name:
        interp_method = method.get('interpolation', 'linear')
        if method.get('group_by_stock', False):
            df[col] = df.groupby('stock')[col].interpolate(method=interp_method)
        else:
            df[col] = df[col].interpolate(method=interp_method)
    
    elif 'Mean Imputation' in method_name:
        if method.get('group_by_stock', False):
            for stock in df['stock'].unique():
                stock_mask = df['stock'] == stock
                mean_val = df.loc[stock_mask, col].mean()
                df.loc[stock_mask & df[col].isnull(), col] = mean_val
        else:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
    
    elif 'Median Imputation' in method_name:
        if method.get('group_by_stock', False):
            for stock in df['stock'].unique():
                stock_mask = df['stock'] == stock
                median_val = df.loc[stock_mask, col].median()
                df.loc[stock_mask & df[col].isnull(), col] = median_val
        else:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    return df

if __name__ == '__main__':
    prepared_df = prepare_data()
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"\nPrepared dataset:")
    print(f"  File: data/integrated_prepared_data.csv")
    print(f"  Rows: {len(prepared_df):,}")
    print(f"  Columns: {len(prepared_df.columns)}")
    print(f"  Null values: {prepared_df.isnull().sum().sum()}")
    
    print("\nSample of prepared data (first row with previously null values):")
    # Show example of handled nulls
    null_cols = ['CPI', 'GDP', 'Unemployment_Rate']
    for col in null_cols:
        if col in prepared_df.columns:
            # Find first row where this column had nulls originally
            sample = prepared_df[prepared_df[col].notna()].iloc[0]
            print(f"\n  {col}:")
            print(f"    Date: {sample['Date']}")
            print(f"    Stock: {sample['stock']}")
            print(f"    Value: {sample[col]}")

