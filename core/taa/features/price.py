"""
Price Feature Generator.
Calculates Momentum, Volatility, and Trend features.
"""

import pandas as pd
import numpy as np
from typing import List
from .base import BaseFeatureGenerator

class PriceFeatureGenerator(BaseFeatureGenerator):
    """
    Generates features based on price history.
    
    Features:
    - Momentum: 1w, 4w, 12w, 52w returns
    - Volatility: 20d, 60d annualized vol
    - Trend: Distance from 200d SMA
    """

    def generate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate price-based features.
        
        Args:
            data: DataFrame with MultiIndex (Ticker, Date) or columns (Ticker, OHLCV).
                  Expects 'Close' column if flattened, or ('Close', 'Ticker') if MultiIndex.
                  
        Returns:
            pd.DataFrame: Feature matrix.
        """
        # Handle MultiIndex (Ticker, Price) from yfinance with group_by='ticker'
        if isinstance(data.columns, pd.MultiIndex):
            # Assuming level 0 is Ticker and level 1 is Price
            # We want to iterate over tickers
            features_list = []
            tickers = data.columns.get_level_values(0).unique()
            
            for ticker in tickers:
                # Extract single ticker data
                # xs returns a DataFrame with Date index and Price columns
                df_ticker = data.xs(ticker, axis=1, level=0).copy()
                
                # Generate features for this ticker
                df_features = self._generate_single_ticker(df_ticker)
                
                # Add Ticker column for later alignment or MultiIndex reconstruction
                df_features['ticker'] = ticker
                features_list.append(df_features)
                
            # Combine all tickers
            if not features_list:
                return pd.DataFrame()
                
            combined_features = pd.concat(features_list)
            
            # Pivot to have Ticker as columns or keep long format?
            # For ML, long format (Date, Ticker) index is usually better.
            # Let's set index to (Date, Ticker)
            combined_features.index.name = 'Date'
            combined_features = combined_features.reset_index().set_index(['Date', 'ticker']).sort_index()
            
            return combined_features
            
        else:
            # Single ticker or flat format
            return self._generate_single_ticker(data)

    def _generate_single_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper to generate features for a single ticker DataFrame.
        Expects 'Close' column.
        """
        if 'Close' not in df.columns:
            return pd.DataFrame()
            
        close = df['Close']
        feats = pd.DataFrame(index=df.index)
        
        # 1. Momentum (Returns)
        # 1 week = 5 trading days
        feats['MOM_1W'] = close.pct_change(5)
        feats['MOM_4W'] = close.pct_change(20)
        feats['MOM_12W'] = close.pct_change(60)
        feats['MOM_52W'] = close.pct_change(252)
        
        # 2. Volatility (Annualized)
        # 20 days (~1 month)
        feats['VOL_20D'] = close.pct_change().rolling(20).std() * np.sqrt(252)
        # 60 days (~3 months)
        feats['VOL_60D'] = close.pct_change().rolling(60).std() * np.sqrt(252)
        
        # 3. Trend (Distance from SMA)
        sma_200 = close.rolling(200).mean()
        feats['DIST_SMA200'] = (close - sma_200) / sma_200
        
        # 4. RSI (14-day)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        feats['RSI_14'] = 100 - (100 / (1 + rs))
        
        return feats
