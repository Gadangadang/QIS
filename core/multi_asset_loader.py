"""
Multi-Asset Data Loader
Loads and aligns price data for multiple futures contracts.
Returns dictionary of DataFrames ready for multi-asset signals.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"


class MultiAssetLoader:
    """
    Loads and aligns price data for multiple assets.
    
    Features:
    - Automatic date alignment across assets
    - Data cleaning (remove invalid rows)
    - Consistent column naming
    - Forward-fill for missing data within gaps
    - Fetch recent data from yfinance if needed
    """
    
    # Map ticker symbols to filenames
    ASSET_FILES = {
        'ES': 'es_futures_2000_2025.csv',
        'NQ': 'nq_futures_2000_2025.csv',
        'GC': 'gc_futures_2000_2025.csv',
        #'SPX': 'spx_1990_2025.csv'
    }
    
    # Map tickers to yfinance symbols
    YFINANCE_SYMBOLS = {
        'ES': 'ES=F',
        'NQ': 'NQ=F',
        'GC': 'GC=F',
    }
    
    def __init__(self, dataset_dir: Optional[Path] = None, use_yfinance: bool = True):
        """
        Initialize loader with dataset directory.
        
        Args:
            dataset_dir: Path to dataset directory
            use_yfinance: If True, fetch recent data from yfinance to supplement CSV files
        """
        self.dataset_dir = dataset_dir or DATASET_DIR
        self.use_yfinance = use_yfinance
        
    def load_single_asset(self, ticker: str) -> pd.DataFrame:
        """
        Load a single asset's price data.
        
        Args:
            ticker: Asset ticker (ES, NQ, GC, SPX)
            
        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume
        """
        if ticker not in self.ASSET_FILES:
            raise ValueError(f"Unknown ticker: {ticker}. Available: {list(self.ASSET_FILES.keys())}")
        
        filepath = self.dataset_dir / self.ASSET_FILES[ticker]
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Load data - handle CSVs with misaligned "Price" column
        # First check if this is a problematic CSV (has Price column)
        header_line = pd.read_csv(filepath, nrows=0)
        has_price_col = 'Price' in [c.strip() for c in header_line.columns]
        
        if has_price_col and len(header_line.columns) > 6:
            # CSV has 7 headers but 6 data columns - skip Price column
            # Read without header, then assign correct column names
            df = pd.read_csv(filepath, skiprows=1, header=None,
                           names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            # Normal CSV
            df = pd.read_csv(filepath, parse_dates=['Date'])
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Remove invalid rows (Close <= 0)
        n_before = len(df)
        df = df[df['Close'] > 0].reset_index(drop=True)
        n_removed = n_before - len(df)
        if n_removed > 0:
            print(f"  {ticker}: Removed {n_removed} rows with Close <= 0")
        
        # Keep only essential columns
        essential_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns:
            essential_cols.append('Volume')
        
        df = df[essential_cols].copy()
        
        # Add ticker column
        df['Ticker'] = ticker
        
        return df
    
    def fetch_yfinance_data(self, ticker: str, start_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Fetch recent data from yfinance.
        
        Args:
            ticker: Asset ticker (ES, NQ, GC)
            start_date: Start date for fetch (typically day after CSV ends)
            
        Returns:
            DataFrame with recent data, or None if fetch fails
        """
        if ticker not in self.YFINANCE_SYMBOLS:
            return None
        
        try:
            import yfinance as yf
            
            yf_symbol = self.YFINANCE_SYMBOLS[ticker]
            yf_ticker = yf.Ticker(yf_symbol)
            
            # Fetch from start_date to today
            end_date = datetime.now()
            data = yf_ticker.history(start=start_date, end=end_date)
            
            if len(data) == 0:
                return None
            
            # Convert to our format
            df = pd.DataFrame({
                'Date': data.index,
                'Open': data['Open'].values,
                'High': data['High'].values,
                'Low': data['Low'].values,
                'Close': data['Close'].values,
                'Volume': data['Volume'].values,
                'Ticker': ticker
            })
            
            # Reset index and ensure Date is datetime (remove timezone)
            df = df.reset_index(drop=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
            return df
            
        except Exception as e:
            print(f"  ⚠️  Failed to fetch yfinance data for {ticker}: {e}")
            return None
    
    def load_assets(
        self, 
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fill_method: str = 'ffill',
        max_fill_days: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple assets and align them to common dates.
        
        Args:
            tickers: List of tickers to load (e.g., ['ES', 'GC'])
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            fill_method: How to fill missing data ('ffill', 'none')
            max_fill_days: Maximum days to forward-fill gaps
            
        Returns:
            Dictionary mapping ticker -> aligned DataFrame
        """
        print(f"\nLoading {len(tickers)} assets: {tickers}")
        print("="*60)
        
        # Load all assets
        raw_data = {}
        for ticker in tickers:
            df = self.load_single_asset(ticker)
            
            # Try to append yfinance data if enabled and CSV data is old
            if self.use_yfinance:
                csv_end_date = df['Date'].max()
                today = pd.Timestamp.now().normalize()
                days_old = (today - csv_end_date).days
                
                if days_old > 7:  # If CSV is more than a week old
                    print(f"  📡 CSV data is {days_old} days old, fetching recent data from yfinance...")
                    yf_data = self.fetch_yfinance_data(ticker, csv_end_date + timedelta(days=1))
                    
                    if yf_data is not None and len(yf_data) > 0:
                        # Append yfinance data to CSV data
                        df = pd.concat([df, yf_data], ignore_index=True)
                        df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
                        print(f"  ✓ Added {len(yf_data)} days from yfinance (now through {df['Date'].max().date()})")
            
            raw_data[ticker] = df
            print(f"✓ {ticker}: {len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # Find common date range
        min_dates = [df['Date'].min() for df in raw_data.values()]
        max_dates = [df['Date'].max() for df in raw_data.values()]
        common_start = max(min_dates)  # Latest start date
        common_end = min(max_dates)    # Earliest end date
        
        print(f"\nCommon date range: {common_start.date()} to {common_end.date()}")
        
        # Apply user-specified date filters
        if start_date:
            common_start = max(common_start, pd.to_datetime(start_date))
        if end_date:
            common_end = min(common_end, pd.to_datetime(end_date))
        
        if start_date or end_date:
            print(f"Filtered date range: {common_start.date()} to {common_end.date()}")
        
        # Filter each asset to common date range
        filtered_data = {}
        for ticker, df in raw_data.items():
            df_filtered = df[(df['Date'] >= common_start) & (df['Date'] <= common_end)].copy()
            filtered_data[ticker] = df_filtered
        
        # Find union of all dates (for alignment)
        all_dates = set()
        for df in filtered_data.values():
            all_dates.update(df['Date'])
        all_dates = sorted(all_dates)
        
        print(f"\nTotal unique dates: {len(all_dates)}")
        
        # Align each asset to the full date range
        aligned_data = {}
        for ticker, df in filtered_data.items():
            # Create full date range DataFrame
            df_full = pd.DataFrame({'Date': all_dates})
            
            # Merge with actual data
            df_aligned = df_full.merge(df, on='Date', how='left')
            
            # Count missing dates
            n_missing = df_aligned['Close'].isna().sum()
            
            # Apply filling strategy
            if fill_method == 'ffill' and n_missing > 0:
                # Forward fill with limit
                df_aligned['Close'] = df_aligned['Close'].ffill(limit=max_fill_days)
                df_aligned['Open'] = df_aligned['Open'].ffill(limit=max_fill_days)
                df_aligned['High'] = df_aligned['High'].ffill(limit=max_fill_days)
                df_aligned['Low'] = df_aligned['Low'].ffill(limit=max_fill_days)
                
                n_filled = n_missing - df_aligned['Close'].isna().sum()
                n_remaining = df_aligned['Close'].isna().sum()
                
                print(f"  {ticker}: {n_missing} missing dates, filled {n_filled}, {n_remaining} remain")
            elif n_missing > 0:
                print(f"  {ticker}: {n_missing} missing dates (no fill)")
            
            # Add ticker column
            df_aligned['Ticker'] = ticker
            
            # Set Date as index
            df_aligned = df_aligned.set_index('Date')
            
            aligned_data[ticker] = df_aligned
        
        # Final statistics
        print(f"\n{'='*60}")
        print("ALIGNMENT SUMMARY")
        print(f"{'='*60}")
        for ticker, df in aligned_data.items():
            n_valid = df['Close'].notna().sum()
            n_total = len(df)
            coverage = n_valid / n_total * 100
            print(f"{ticker}: {n_valid}/{n_total} valid dates ({coverage:.1f}%)")
        
        return aligned_data
    
    def get_common_dates_only(self, aligned_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Filter aligned data to only dates where ALL assets have valid data.
        
        Args:
            aligned_data: Dictionary of aligned DataFrames (from load_assets)
            
        Returns:
            Dictionary with only common valid dates
        """
        # Find dates valid for all assets
        valid_dates = None
        for ticker, df in aligned_data.items():
            dates_with_data = set(df[df['Close'].notna()]['Date'])
            if valid_dates is None:
                valid_dates = dates_with_data
            else:
                valid_dates &= dates_with_data
        
        valid_dates = sorted(valid_dates)
        
        # Filter each asset
        common_data = {}
        for ticker, df in aligned_data.items():
            df_common = df[df['Date'].isin(valid_dates)].reset_index(drop=True)
            common_data[ticker] = df_common
        
        print(f"\nFiltered to {len(valid_dates)} common dates")
        return common_data


def load_assets(tickers: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load and align multiple assets.
    Automatically fetches recent data from yfinance if CSV files are outdated.
    
    Args:
        tickers: List of tickers (e.g., ['ES', 'GC', 'NQ'])
        **kwargs: Additional arguments for MultiAssetLoader.load_assets()
            - start_date: Optional start date filter
            - end_date: Optional end date filter
            - use_yfinance: If True (default), fetch recent data from yfinance
        
    Returns:
        Dictionary mapping ticker -> aligned DataFrame
    """
    use_yfinance = kwargs.pop('use_yfinance', True)
    loader = MultiAssetLoader(use_yfinance=use_yfinance)
    return loader.load_assets(tickers, **kwargs)


if __name__ == "__main__":
    # Test the loader
    print("Testing MultiAssetLoader...")
    
    # Load ES + GC
    data = load_assets(['ES', 'GC'], start_date='2010-01-01')
    
    # Show sample
    print("\nSample data:")
    for ticker, df in data.items():
        print(f"\n{ticker}:")
        print(df.head(3))
    
    # Check alignment
    print("\nDate alignment check:")
    es_dates = set(data['ES']['Date'])
    gc_dates = set(data['GC']['Date'])
    print(f"ES dates: {len(es_dates)}")
    print(f"GC dates: {len(gc_dates)}")
    print(f"Intersection: {len(es_dates & gc_dates)}")
    print(f"ES only: {len(es_dates - gc_dates)}")
    print(f"GC only: {len(gc_dates - es_dates)}")
    
    # Get common dates only
    loader = MultiAssetLoader()
    common_data = loader.get_common_dates_only(data)
    
    print("\nCommon dates only:")
    for ticker, df in common_data.items():
        print(f"{ticker}: {len(df)} rows, all valid")
