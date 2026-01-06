"""
TAA Value Signals for Valuation-Based Asset Allocation.

Value signals identify assets trading at attractive prices relative to
fundamentals, historical norms, or cross-sectional comparisons.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Union
from signals.base import SignalModel


class CAPESignal(SignalModel):
    """
    Cyclically-Adjusted Price-to-Earnings (CAPE) ratio signal for equities.
    
    Also known as Shiller P/E ratio. Compares current price to 10-year
    average earnings (inflation-adjusted). Low CAPE suggests undervaluation.
    
    Args:
        normalization: Method for signal generation ('zscore', 'percentile', 'spread')
        lookback_years: Years of history for z-score/percentile calculation (default: 20)
        invert: If True, low CAPE = positive signal (default: True)
    
    Returns:
        DataFrame with 'Signal' column (value-based expected return)
    
    Example:
        >>> cape = CAPESignal(normalization='zscore', invert=True)
        >>> # DataFrame needs 'CAPE' column (from external data)
        >>> signals = cape.generate(equity_data)
    
    Note:
        Input DataFrame must have 'CAPE' column. Calculate externally as:
        CAPE = Price / avg(Real_Earnings_10Y)
    """
    
    def __init__(
        self,
        normalization: str = 'zscore',
        lookback_years: int = 20,
        invert: bool = True
    ):
        if normalization not in ['zscore', 'percentile', 'spread']:
            raise ValueError(f"normalization must be 'zscore', 'percentile', or 'spread', got {normalization}")
        
        self.normalization = normalization
        self.lookback_years = lookback_years
        self.invert = invert
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate CAPE-based value signals.
        
        Args:
            df: DataFrame with 'CAPE' column
        
        Returns:
            DataFrame with 'Signal' column
        """
        df = df.copy()
        
        if 'CAPE' not in df.columns:
            raise ValueError("DataFrame must have 'CAPE' column")
        
        # Calculate lookback window in periods (assume monthly data)
        lookback_periods = self.lookback_years * 12
        
        # Generate signal based on normalization method
        if self.normalization == 'zscore':
            # Z-score relative to historical mean
            rolling_mean = df['CAPE'].rolling(window=lookback_periods, min_periods=12).mean()
            rolling_std = df['CAPE'].rolling(window=lookback_periods, min_periods=12).std()
            df['CAPE_ZScore'] = (df['CAPE'] - rolling_mean) / rolling_std
            
            # Invert: high CAPE (expensive) = negative signal
            df['Signal'] = -df['CAPE_ZScore'] if self.invert else df['CAPE_ZScore']
        
        elif self.normalization == 'percentile':
            # Percentile rank (0-1)
            df['CAPE_Percentile'] = df['CAPE'].rolling(
                window=lookback_periods, 
                min_periods=12
            ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            
            # Invert: low percentile (cheap) = positive signal
            df['Signal'] = 1 - df['CAPE_Percentile'] if self.invert else df['CAPE_Percentile']
        
        elif self.normalization == 'spread':
            # Deviation from long-term average (%)
            rolling_mean = df['CAPE'].rolling(window=lookback_periods, min_periods=12).mean()
            df['CAPE_Spread'] = (df['CAPE'] - rolling_mean) / rolling_mean
            
            # Invert: negative spread (below average) = positive signal
            df['Signal'] = -df['CAPE_Spread'] if self.invert else df['CAPE_Spread']
        
        # Warm-up period
        df.iloc[:lookback_periods, df.columns.get_loc('Signal')] = np.nan
        
        return df


class RealYieldSpread(SignalModel):
    """
    Real Yield Spread signal for bonds.
    
    Compares real yields (nominal - inflation) to historical averages or
    across asset classes. Higher real yields suggest better value.
    
    Args:
        comparison: 'historical' or 'cross_sectional'
        lookback_years: Years for historical comparison (default: 10)
    
    Returns:
        DataFrame with 'Signal' column (yield attractiveness)
    
    Example:
        >>> spread = RealYieldSpread(comparison='historical')
        >>> # Needs 'NominalYield' and 'InflationExpectation' columns
        >>> signals = spread.generate(bond_data)
    """
    
    def __init__(
        self,
        comparison: str = 'historical',
        lookback_years: int = 10
    ):
        if comparison not in ['historical', 'cross_sectional']:
            raise ValueError(f"comparison must be 'historical' or 'cross_sectional', got {comparison}")
        
        self.comparison = comparison
        self.lookback_years = lookback_years
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate real yield spread signals."""
        df = df.copy()
        
        # Calculate real yield
        if 'RealYield' not in df.columns:
            if 'NominalYield' in df.columns and 'InflationExpectation' in df.columns:
                df['RealYield'] = df['NominalYield'] - df['InflationExpectation']
            elif 'Yield' in df.columns and 'InflationExpectation' in df.columns:
                df['RealYield'] = df['Yield'] - df['InflationExpectation']
            else:
                raise ValueError(
                    "DataFrame must have 'RealYield' or ('NominalYield'/'Yield' and 'InflationExpectation') columns"
                )
        
        lookback_periods = self.lookback_years * 12
        
        if self.comparison == 'historical':
            # Compare to historical average
            rolling_mean = df['RealYield'].rolling(window=lookback_periods, min_periods=12).mean()
            rolling_std = df['RealYield'].rolling(window=lookback_periods, min_periods=12).std()
            
            # Z-score (higher real yield = positive signal)
            df['Signal'] = (df['RealYield'] - rolling_mean) / rolling_std
        
        else:  # cross_sectional
            # For cross-sectional, just return real yield (normalize externally)
            df['Signal'] = df['RealYield']
        
        # Warm-up
        df.iloc[:lookback_periods, df.columns.get_loc('Signal')] = np.nan
        
        return df


class EarningsYieldSignal(SignalModel):
    """
    Earnings Yield (E/P) signal for equity valuation.
    
    Compares earnings yield to bond yields (Fed Model) or historical norms.
    Higher earnings yield relative to bonds suggests equity attractiveness.
    
    Args:
        comparison: 'absolute', 'vs_bonds', or 'zscore'
        lookback_years: Years for z-score calculation (default: 15)
    
    Returns:
        DataFrame with 'Signal' column
    
    Example:
        >>> ey = EarningsYieldSignal(comparison='vs_bonds')
        >>> # Needs 'EarningsYield' and optionally 'BondYield' columns
        >>> signals = ey.generate(equity_data)
    
    Note:
        EarningsYield = Earnings / Price = 1 / PE_Ratio
    """
    
    def __init__(
        self,
        comparison: str = 'zscore',
        lookback_years: int = 15
    ):
        if comparison not in ['absolute', 'vs_bonds', 'zscore']:
            raise ValueError(f"comparison must be 'absolute', 'vs_bonds', or 'zscore', got {comparison}")
        
        self.comparison = comparison
        self.lookback_years = lookback_years
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate earnings yield signals."""
        df = df.copy()
        
        # Calculate earnings yield if not provided
        if 'EarningsYield' not in df.columns:
            if 'PE_Ratio' in df.columns:
                df['EarningsYield'] = 1 / df['PE_Ratio']
            elif 'Earnings' in df.columns and 'Price' in df.columns:
                df['EarningsYield'] = df['Earnings'] / df['Price']
            else:
                raise ValueError(
                    "DataFrame must have 'EarningsYield' or 'PE_Ratio' or ('Earnings' and 'Price') columns"
                )
        
        lookback_periods = self.lookback_years * 12
        
        if self.comparison == 'absolute':
            # Raw earnings yield as signal
            df['Signal'] = df['EarningsYield']
        
        elif self.comparison == 'vs_bonds':
            # Equity risk premium (earnings yield - bond yield)
            if 'BondYield' not in df.columns:
                raise ValueError("comparison='vs_bonds' requires 'BondYield' column")
            
            df['EquityRiskPremium'] = df['EarningsYield'] - df['BondYield']
            df['Signal'] = df['EquityRiskPremium']
        
        elif self.comparison == 'zscore':
            # Z-score of earnings yield
            rolling_mean = df['EarningsYield'].rolling(window=lookback_periods, min_periods=12).mean()
            rolling_std = df['EarningsYield'].rolling(window=lookback_periods, min_periods=12).std()
            
            df['Signal'] = (df['EarningsYield'] - rolling_mean) / rolling_std
            
            # Warm-up only for zscore
            df.iloc[:lookback_periods, df.columns.get_loc('Signal')] = np.nan
        
        return df


class RelativeValueSignal(SignalModel):
    """
    Relative Value signal using z-scores of valuation metrics.
    
    Generic signal generator for any valuation metric (P/B, P/S, Div Yield, etc.)
    relative to historical norms. Useful for commodities, currencies, and alternatives.
    
    Args:
        metric_column: Name of column containing valuation metric
        lookback_years: Years for z-score calculation (default: 10)
        invert: If True, high metric = negative signal (for P/E, P/B, etc.) (default: True)
        outlier_clip: Clip z-scores beyond this threshold (default: 3.0)
    
    Returns:
        DataFrame with 'Signal' column
    
    Example:
        >>> # For Price-to-Book ratio
        >>> pb_signal = RelativeValueSignal(metric_column='PB_Ratio', invert=True)
        >>> signals = pb_signal.generate(stock_data)
        >>> 
        >>> # For Dividend Yield
        >>> div_signal = RelativeValueSignal(metric_column='DivYield', invert=False)
        >>> signals = div_signal.generate(stock_data)
    """
    
    def __init__(
        self,
        metric_column: str,
        lookback_years: int = 10,
        invert: bool = True,
        outlier_clip: float = 3.0
    ):
        self.metric_column = metric_column
        self.lookback_years = lookback_years
        self.invert = invert
        self.outlier_clip = outlier_clip
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate relative value signals."""
        df = df.copy()
        
        if self.metric_column not in df.columns:
            raise ValueError(f"DataFrame must have '{self.metric_column}' column")
        
        lookback_periods = self.lookback_years * 12
        
        # Calculate z-score
        rolling_mean = df[self.metric_column].rolling(window=lookback_periods, min_periods=12).mean()
        rolling_std = df[self.metric_column].rolling(window=lookback_periods, min_periods=12).std()
        
        df['Value_ZScore'] = (df[self.metric_column] - rolling_mean) / rolling_std
        
        # Clip outliers
        if self.outlier_clip is not None:
            df['Value_ZScore'] = df['Value_ZScore'].clip(-self.outlier_clip, self.outlier_clip)
        
        # Invert if needed (high P/E = expensive = negative signal)
        df['Signal'] = -df['Value_ZScore'] if self.invert else df['Value_ZScore']
        
        # Warm-up
        df.iloc[:lookback_periods, df.columns.get_loc('Signal')] = np.nan
        
        return df


class CommodityValueSignal(SignalModel):
    """
    Commodity-specific value signal using price-to-marginal-cost or inventory levels.
    
    For commodities, value is assessed via:
    - Price vs. production cost
    - Inventory levels (high inventory = oversupply = bearish)
    - Price vs. long-term trend
    
    Args:
        method: 'price_trend', 'inventory', or 'cost_spread'
        lookback_years: Years for trend/z-score calculation (default: 5)
    
    Returns:
        DataFrame with 'Signal' column
    
    Example:
        >>> oil_value = CommodityValueSignal(method='price_trend')
        >>> signals = oil_value.generate(crude_oil_data)
    """
    
    def __init__(
        self,
        method: str = 'price_trend',
        lookback_years: int = 5
    ):
        if method not in ['price_trend', 'inventory', 'cost_spread']:
            raise ValueError(f"method must be 'price_trend', 'inventory', or 'cost_spread', got {method}")
        
        self.method = method
        self.lookback_years = lookback_years
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate commodity value signals."""
        df = df.copy()
        
        lookback_periods = self.lookback_years * 12
        
        if self.method == 'price_trend':
            # Price deviation from long-term trend
            if 'Close' not in df.columns:
                raise ValueError("method='price_trend' requires 'Close' column")
            
            rolling_mean = df['Close'].rolling(window=lookback_periods, min_periods=12).mean()
            rolling_std = df['Close'].rolling(window=lookback_periods, min_periods=12).std()
            
            # Negative z-score = below trend = cheap = positive signal
            df['Signal'] = -(df['Close'] - rolling_mean) / rolling_std
        
        elif self.method == 'inventory':
            # Inventory levels (low inventory = bullish)
            if 'Inventory' not in df.columns:
                raise ValueError("method='inventory' requires 'Inventory' column")
            
            rolling_mean = df['Inventory'].rolling(window=lookback_periods, min_periods=12).mean()
            rolling_std = df['Inventory'].rolling(window=lookback_periods, min_periods=12).std()
            
            # Negative z-score = low inventory = bullish
            df['Signal'] = -(df['Inventory'] - rolling_mean) / rolling_std
        
        elif self.method == 'cost_spread':
            # Price vs. marginal cost of production
            if 'Price' not in df.columns or 'MarginalCost' not in df.columns:
                raise ValueError("method='cost_spread' requires 'Price' and 'MarginalCost' columns")
            
            df['CostSpread'] = df['Price'] / df['MarginalCost'] - 1
            
            rolling_mean = df['CostSpread'].rolling(window=lookback_periods, min_periods=12).mean()
            rolling_std = df['CostSpread'].rolling(window=lookback_periods, min_periods=12).std()
            
            # Negative z-score = trading near cost = bullish
            df['Signal'] = -(df['CostSpread'] - rolling_mean) / rolling_std
        
        # Warm-up
        df.iloc[:lookback_periods, df.columns.get_loc('Signal')] = np.nan
        
        return df


class CrossSectionalValue(SignalModel):
    """
    Cross-sectional value signal comparing assets within a universe.
    
    Ranks assets by valuation metric and generates relative value signals.
    Must be applied across multiple assets simultaneously.
    
    Args:
        metric_column: Valuation metric to rank (e.g., 'PE_Ratio', 'PB_Ratio', 'DivYield')
        invert_rank: If True, low metric = high rank (for P/E, P/B) (default: True)
    
    Returns:
        DataFrame with 'Signal' column (percentile rank)
    
    Example:
        >>> # For individual asset, just calculates metric
        >>> cs_value = CrossSectionalValue(metric_column='PE_Ratio', invert_rank=True)
        >>> signals = cs_value.generate(stock_data)
        >>> 
        >>> # For proper cross-sectional, use with MultiAssetEnsemble
    
    Note:
        For true cross-sectional signals, use this with MultiAssetEnsemble
        which handles normalization across assets.
    """
    
    def __init__(
        self,
        metric_column: str,
        invert_rank: bool = True
    ):
        self.metric_column = metric_column
        self.invert_rank = invert_rank
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cross-sectional value signal for single asset.
        
        Returns raw metric. Cross-sectional normalization happens
        in MultiAssetEnsemble.
        """
        df = df.copy()
        
        if self.metric_column not in df.columns:
            raise ValueError(f"DataFrame must have '{self.metric_column}' column")
        
        # For single asset, just return metric (will be normalized cross-sectionally)
        if self.invert_rank:
            # Invert: low P/E = high value = positive signal
            df['Signal'] = -df[self.metric_column]
        else:
            # Don't invert: high Div Yield = high value = positive signal
            df['Signal'] = df[self.metric_column]
        
        return df
