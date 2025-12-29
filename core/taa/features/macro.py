"""
Macro Feature Generator.
Integrates FRED data (Yields, Spreads, VIX).
"""

import pandas as pd
from .base import BaseFeatureGenerator

class MacroFeatureGenerator(BaseFeatureGenerator):
    """
    Generates macro-economic features.
    These are usually "global" features common to all assets.
    """

    def generate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate macro features.
        
        Args:
            data: DataFrame containing FRED data (columns = series IDs).
            
        Returns:
            pd.DataFrame: Feature matrix with macro indicators.
        """
        if data.empty:
            return pd.DataFrame()
            
        feats = pd.DataFrame(index=data.index)
        
        # 1. Yield Curve Slope (10Y - 2Y)
        # Series IDs: T10Y2Y is often directly available, or calculate from DGS10 - DGS2
        if 'T10Y2Y' in data.columns:
            feats['YIELD_CURVE_SLOPE'] = data['T10Y2Y']
        elif 'DGS10' in data.columns and 'DGS2' in data.columns:
            feats['YIELD_CURVE_SLOPE'] = data['DGS10'] - data['DGS2']
            
        # 2. Credit Spread (BAA - 10Y Treasury)
        # Series IDs: BAMLC0A0CM (US Corp Master Option-Adjusted Spread) or similar
        # Or BAA10Y (Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity)
        if 'BAA10Y' in data.columns:
            feats['CREDIT_SPREAD'] = data['BAA10Y']
            
        # 3. VIX (Volatility Index)
        # Series ID: VIXCLS
        if 'VIXCLS' in data.columns:
            feats['VIX'] = data['VIXCLS']
            
        # 4. Inflation Trend (CPI YoY)
        # Series ID: CPIAUCSL (Consumer Price Index for All Urban Consumers: All Items in U.S. City Average)
        if 'CPIAUCSL' in data.columns:
            # CPI is monthly, calculate YoY change
            feats['CPI_YOY'] = data['CPIAUCSL'].pct_change(12)
            
        # Forward fill macro data to handle weekends/holidays if necessary
        # (Though usually we align to the asset's index later)
        return feats.ffill()
