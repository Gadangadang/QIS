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
            features_list = []
            
            # Determine which level contains the Tickers
            # yfinance typically puts Tickers at level 0 if group_by='ticker'
            # but level 1 if you downloaded multiple assets without that flag.
            ticker_level = 0 
            if 'Close' in data.columns.get_level_values(0):
                ticker_level = 1 # Tickers are actually in level 1
                
            tickers = data.columns.get_level_values(ticker_level).unique()
            
            for ticker in tickers:
                try:
                    # Use .xs with the correct level we just found
                    df_ticker = data.xs(ticker, axis=1, level=ticker_level).copy()
                    
                    # Double check the ticker data isn't empty
                    if df_ticker.empty:
                        continue
                        
                    df_features = self._generate_single_ticker(df_ticker)
                    
                    # Only append if features were actually created
                    if df_features is not None and not df_features.empty:
                        df_features['ticker'] = ticker
                        features_list.append(df_features)
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
                    continue
                    
            # Combine all tickers
            if not features_list:
                return pd.DataFrame()
                
            combined_features = pd.concat(features_list)
            
            current_index_name = combined_features.index.name or 'date'
            
            # Reset index and set the new MultiIndex (using the actual name found)
            combined_features = combined_features.reset_index()
            combined_features = combined_features.set_index([current_index_name, 'ticker']).sort_index()
            
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
            
        # FIX: Ensure 'close' is a Series, even if df['Close'] returns a DataFrame
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0] # Take the first 'Close' column
        
        feats = pd.DataFrame(index=df.index)
        
        # Now pct_change will return a Series, matching the column 'MOM_1W'
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
