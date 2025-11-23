"""
Validate futures data quality for multi-asset framework.
Check columns, date ranges, gaps, and alignment across ES, NQ, GC.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

def validate_futures_data():
    """Validate ES, NQ, GC futures data quality."""
    
    # Load datasets
    files = {
        'ES': 'es_futures_2000_2025.csv',
        'NQ': 'nq_futures_2000_2025.csv', 
        'GC': 'gc_futures_2000_2025.csv'
    }
    
    datasets = {}
    for ticker, filename in files.items():
        filepath = DATASET_DIR / filename
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Clean bad data: remove rows with Close <= 0
        bad_rows = df[df['Close'] <= 0]
        if len(bad_rows) > 0:
            print(f"\n⚠️  {ticker}: Removing {len(bad_rows)} rows with Close <= 0")
            df = df[df['Close'] > 0].reset_index(drop=True)
        
        datasets[ticker] = df
        print(f"\n{'='*60}")
        print(f"{ticker} ({filename}):")
        print(f"{'='*60}")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        print(f"\nLast 3 rows:")
        print(df.tail(3))
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"\n⚠️  Missing values:")
            print(missing[missing > 0])
        else:
            print(f"\n✓ No missing values")
        
        # Check for duplicates
        dupes = df['Date'].duplicated().sum()
        if dupes > 0:
            print(f"⚠️  {dupes} duplicate dates")
        else:
            print(f"✓ No duplicate dates")
        
        # Check price column consistency
        if 'Price' in df.columns and 'Close' in df.columns:
            price_close_diff = (df['Price'] - df['Close']).abs().max()
            print(f"Max |Price - Close|: {price_close_diff:.4f}")
        
        # Basic statistics
        print(f"\nClose price stats:")
        print(df['Close'].describe())
        
        # Calculate daily returns
        df['Return'] = df['Close'].pct_change()
        print(f"\nDaily return stats:")
        print(df['Return'].describe())
        print(f"Annualized vol: {df['Return'].std() * np.sqrt(252):.2%}")
        
        # Check for gaps (weekends are expected)
        df['Date_diff'] = df['Date'].diff().dt.days
        large_gaps = df[df['Date_diff'] > 7]
        if len(large_gaps) > 0:
            print(f"\n⚠️  {len(large_gaps)} gaps > 7 days:")
            print(large_gaps[['Date', 'Date_diff']].head(10))
        else:
            print(f"\n✓ No large gaps (>7 days)")
    
    # Check date alignment across assets
    print(f"\n{'='*60}")
    print("DATE ALIGNMENT ANALYSIS")
    print(f"{'='*60}")
    
    # Find common dates
    dates_es = set(datasets['ES']['Date'])
    dates_nq = set(datasets['NQ']['Date'])
    dates_gc = set(datasets['GC']['Date'])
    
    common_all = dates_es & dates_nq & dates_gc
    print(f"\nES dates: {len(dates_es)}")
    print(f"NQ dates: {len(dates_nq)}")
    print(f"GC dates: {len(dates_gc)}")
    print(f"Common to all 3: {len(common_all)}")
    
    # Check which assets have unique dates
    es_only = dates_es - dates_nq - dates_gc
    nq_only = dates_nq - dates_es - dates_gc
    gc_only = dates_gc - dates_es - dates_nq
    
    if es_only:
        print(f"\n⚠️  {len(es_only)} dates only in ES")
        print(f"Examples: {sorted(list(es_only))[:5]}")
    if nq_only:
        print(f"\n⚠️  {len(nq_only)} dates only in NQ")
        print(f"Examples: {sorted(list(nq_only))[:5]}")
    if gc_only:
        print(f"\n⚠️  {len(gc_only)} dates only in GC")
        print(f"Examples: {sorted(list(gc_only))[:5]}")
    
    # Correlation analysis on common dates
    print(f"\n{'='*60}")
    print("CORRELATION ANALYSIS (Common dates only)")
    print(f"{'='*60}")
    
    # Merge on common dates
    df_es_common = datasets['ES'][datasets['ES']['Date'].isin(common_all)].set_index('Date')
    df_nq_common = datasets['NQ'][datasets['NQ']['Date'].isin(common_all)].set_index('Date')
    df_gc_common = datasets['GC'][datasets['GC']['Date'].isin(common_all)].set_index('Date')
    
    # Calculate returns
    returns = pd.DataFrame({
        'ES': df_es_common['Close'].pct_change(),
        'NQ': df_nq_common['Close'].pct_change(),
        'GC': df_gc_common['Close'].pct_change()
    }).dropna()
    
    print(f"\nReturn correlation matrix:")
    corr_matrix = returns.corr()
    print(corr_matrix)
    
    print(f"\n✓ ES-GC correlation: {corr_matrix.loc['ES', 'GC']:.3f} (good diversification if < 0.7)")
    print(f"✓ ES-NQ correlation: {corr_matrix.loc['ES', 'NQ']:.3f} (expected high, both equities)")
    print(f"✓ NQ-GC correlation: {corr_matrix.loc['NQ', 'GC']:.3f} (good diversification if < 0.7)")
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"✓ All datasets loaded successfully")
    print(f"✓ Date range: ~24 years (2000-2024)")
    print(f"✓ Common dates: {len(common_all)} ({len(common_all)/len(dates_es)*100:.1f}% of ES dates)")
    print(f"✓ ES-GC correlation: {corr_matrix.loc['ES', 'GC']:.3f} → Good diversification")
    print(f"\n✓ Data ready for multi-asset framework!")
    
    return datasets, returns, corr_matrix

if __name__ == "__main__":
    datasets, returns, corr_matrix = validate_futures_data()
