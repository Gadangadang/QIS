"""
Relative Value Feature Generator.
Calculates features relative to a benchmark (e.g., ACWI or SPY).
"""

import pandas as pd
import numpy as np
from .base import BaseFeatureGenerator

class RelativeValueFeatureGenerator(BaseFeatureGenerator):
    def generate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        benchmark = kwargs.get('benchmark')
        if benchmark is None:
            raise ValueError("Benchmark DataFrame is required for relative features.")

        # 1. Extract and Clean Benchmark Close
        bench_close = self.extract_close(benchmark)
        
        # Ensure benchmark index is datetime
        if not pd.api.types.is_datetime64_any_dtype(bench_close.index):
            bench_close.index = pd.to_datetime(bench_close.index)
        
        # Squeeze to Series
        bench_close = bench_close.squeeze()

        features_list = []
        
        if isinstance(data.columns, pd.MultiIndex):
            tickers = data.columns.get_level_values(0).unique()
            for ticker in tickers:
                # 2. Extract and Clean Asset Close
                asset_close = self.extract_close(data, ticker)
                
                # Ensure asset index is datetime
                if not pd.api.types.is_datetime64_any_dtype(asset_close.index):
                    asset_close.index = pd.to_datetime(asset_close.index)
                
                asset_close = asset_close.squeeze()

                # 3. Align both on the index (inner join ensures matching dates)
                aligned_asset, aligned_bench = asset_close.align(bench_close, join='inner', axis=0)
                
                # Safety check: if alignment results in empty data
                if aligned_asset.empty:
                    continue

                # 4. Calculation Logic
                ratio = aligned_asset / aligned_bench
                
                df_features = pd.DataFrame(index=aligned_asset.index)
                df_features['REL_MOM_4W'] = ratio.pct_change(20)
                df_features['REL_MOM_12W'] = ratio.pct_change(60)
                
                # RSI on Ratio
                delta = ratio.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                # Avoid division by zero
                rs = gain / loss.replace(0, 1e-9)
                df_features['REL_RSI'] = 100 - (100 / (1 + rs))
                
                # Beta Calculation
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
        
        # Ensure the index name is consistent for the final MultiIndex
        combined_features.index.name = 'date'
        return combined_features.reset_index().set_index(['date', 'ticker']).sort_index()

    def extract_close(self, df, ticker=None):
        """Standardized extraction helper."""
        if isinstance(df.columns, pd.MultiIndex):
            if ticker is None:
                ticker = df.columns.get_level_values(0)[0]
            # Use .iloc[:, 0] to force a Series if duplicates exist
            res = df.xs(ticker, axis=1, level=0)['Close']
        else:
            res = df['Close']
            
        return res.iloc[:, 0] if isinstance(res, pd.DataFrame) else res