"""
Relative Value Feature Generator.
Calculates features relative to a benchmark (e.g., ACWI or SPY).
"""

import pandas as pd
import numpy as np
from .base import BaseFeatureGenerator

class RelativeValueFeatureGenerator(BaseFeatureGenerator):
    """
    Generates features comparing the asset to a benchmark.
    """

    def generate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate relative value features.
        
        Args:
            data: DataFrame with MultiIndex (Price, Ticker) for assets.
            **kwargs: Must contain 'benchmark' DataFrame (single ticker).
            
        Returns:
            pd.DataFrame: Feature matrix (Date, Ticker) index.
        """
        benchmark = kwargs.get('benchmark')
        if benchmark is None or benchmark.empty:
            return pd.DataFrame()
            
        # Ensure benchmark has 'Close'
        if 'Close' not in benchmark.columns:
            # Try to find close in multiindex if it exists
            if isinstance(benchmark.columns, pd.MultiIndex):
                 # Assume single ticker in benchmark, take first level 0 (Ticker)
                 ticker = benchmark.columns.get_level_values(0)[0]
                 bench_close = benchmark.xs(ticker, axis=1, level=0)['Close']
            else:
                return pd.DataFrame()
        else:
            bench_close = benchmark['Close']

        # Align benchmark to data index
        # (We assume data is the master index provider or we align both)
        
        features_list = []
        
        if isinstance(data.columns, pd.MultiIndex):
            tickers = data.columns.get_level_values(0).unique()
            
            for ticker in tickers:
                asset_close = data.xs(ticker, axis=1, level=0)['Close']
                
                # Align dates
                aligned_asset, aligned_bench = asset_close.align(bench_close, join='inner')
                
                # Calculate Relative Price Ratio
                ratio = aligned_asset / aligned_bench
                
                df_features = pd.DataFrame(index=aligned_asset.index)
                
                # 1. Relative Momentum
                df_features['REL_MOM_4W'] = ratio.pct_change(20)
                df_features['REL_MOM_12W'] = ratio.pct_change(60)
                
                # 2. Relative Strength (RSI of the Ratio)
                delta = ratio.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df_features['REL_RSI'] = 100 - (100 / (1 + rs))
                
                # 3. Beta (Rolling 60d)
                # Covariance(asset_ret, bench_ret) / Var(bench_ret)
                asset_ret = aligned_asset.pct_change()
                bench_ret = aligned_bench.pct_change()
                
                cov = asset_ret.rolling(60).cov(bench_ret)
                var = bench_ret.rolling(60).var()
                df_features['BETA_60D'] = cov / var
                
                df_features['ticker'] = ticker
                features_list.append(df_features)
                
        if not features_list:
            return pd.DataFrame()
            
        combined_features = pd.concat(features_list)
        return combined_features.reset_index().set_index(['Date', 'ticker']).sort_index()
