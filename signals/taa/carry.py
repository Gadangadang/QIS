"""
TAA Carry Signals for Yield-Based Strategies.

Carry signals capture return from holding assets that pay income
(bond yields, dividend yields, commodity roll yield).
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from signals.base import SignalModel


class YieldCarry(SignalModel):
    """
    Yield Carry signal for TAA using bond yields and dividend yields.
    
    Combines Treasury yield data (from FRED) with equity dividend yields
    (from Yahoo Finance) to generate carry-based expected return forecasts.
    
    Strategy:
    - Higher yields → Higher expected returns (carry)
    - Yield spreads vs. risk-free rate
    - Real yields (inflation-adjusted)
    
    Args:
        yield_type: 'absolute', 'spread', or 'real' (default: 'spread')
        risk_free_rate: Annual risk-free rate for spread calculation (default: 0.02)
        inflation_adjust: Adjust for inflation expectations (default: False)
        min_yield: Minimum yield to generate signal (default: 0.0)
    
    Returns:
        DataFrame with 'Signal' column containing yield-based expected returns
    
    Example:
        >>> carry = YieldCarry(yield_type='spread')
        >>> # For bonds, provide 'Yield' column
        >>> bond_signals = carry.generate(bond_data)
        >>> # For stocks, calculate dividend yield first
        >>> stock_data['Yield'] = stock_data['DividendYield']
        >>> stock_signals = carry.generate(stock_data)
    
    Note:
        Input DataFrame must have either:
        - 'Yield' column (for bonds/dividend yields), or
        - 'Close' and 'Dividend' columns (will calculate yield)
    """
    
    def __init__(
        self,
        yield_type: str = 'spread',
        risk_free_rate: float = 0.02,
        inflation_adjust: bool = False,
        min_yield: float = 0.0
    ):
        if yield_type not in ['absolute', 'spread', 'real']:
            raise ValueError(f"yield_type must be 'absolute', 'spread', or 'real', got {yield_type}")
        
        self.yield_type = yield_type
        self.risk_free_rate = risk_free_rate
        self.inflation_adjust = inflation_adjust
        self.min_yield = min_yield
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate yield carry signals.
        
        Args:
            df: DataFrame with 'Yield' column OR 'Close' + 'Dividend' columns
        
        Returns:
            DataFrame with 'Signal' column (yield-based expected return)
        """
        df = df.copy()
        
        # Calculate yield if not provided
        if 'Yield' not in df.columns:
            if 'Dividend' in df.columns and 'Close' in df.columns:
                # Annualized dividend yield
                df['Yield'] = (df['Dividend'] / df['Close']) * 100  # In percent
            else:
                raise ValueError(
                    "DataFrame must have 'Yield' column or both 'Close' and 'Dividend' columns"
                )
        
        # Ensure yield is numeric and non-negative
        df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce')
        df['Yield'] = df['Yield'].clip(lower=0)
        
        # Calculate signal based on yield_type
        if self.yield_type == 'absolute':
            # Raw yield as signal
            df['Signal'] = df['Yield'] / 100  # Convert to decimal
        
        elif self.yield_type == 'spread':
            # Yield spread over risk-free rate
            df['Signal'] = (df['Yield'] / 100) - self.risk_free_rate
        
        elif self.yield_type == 'real':
            # Real yield (need inflation expectations)
            if 'InflationExpectation' in df.columns:
                # Inflation-adjusted yield
                df['Signal'] = (df['Yield'] / 100) - (df['InflationExpectation'] / 100)
            else:
                # Fallback: use constant inflation assumption (2%)
                df['Signal'] = (df['Yield'] / 100) - 0.02
        
        # Apply minimum yield filter
        df.loc[df['Yield'] < self.min_yield, 'Signal'] = 0
        
        # Forward fill yields (monthly data may have gaps)
        df['Signal'] = df['Signal'].ffill()
        
        return df


class RollYield(SignalModel):
    """
    Roll Yield signal for futures-based TAA strategies.
    
    Captures return from rolling futures contracts in contango/backwardation.
    Positive roll yield (backwardation) is bullish, negative (contango) is bearish.
    
    Args:
        term_structure_window: Window to estimate term structure slope (default: 3)
        annualize: Annualize roll yield (default: True)
    
    Returns:
        DataFrame with 'Signal' column containing annualized roll yield
    
    Example:
        >>> roll = RollYield(term_structure_window=3)
        >>> # Requires 'FrontMonth' and 'NextMonth' futures prices
        >>> signals = roll.generate(futures_data)
    
    Note:
        Input DataFrame must have:
        - 'FrontMonth': Front-month futures price
        - 'NextMonth': Next-month futures price
        - DatetimeIndex with frequency information
    """
    
    def __init__(
        self,
        term_structure_window: int = 3,
        annualize: bool = True
    ):
        self.term_structure_window = term_structure_window
        self.annualize = annualize
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate roll yield signals."""
        df = df.copy()
        
        # Validate required columns
        if 'FrontMonth' not in df.columns or 'NextMonth' not in df.columns:
            raise ValueError(
                "DataFrame must have 'FrontMonth' and 'NextMonth' columns for roll yield calculation"
            )
        
        # Calculate term structure slope (backwardation < 0, contango > 0)
        df['TermStructure'] = (df['NextMonth'] / df['FrontMonth'] - 1) * 100
        
        # Smooth with rolling average
        df['TermStructure_Smooth'] = df['TermStructure'].rolling(
            window=self.term_structure_window
        ).mean()
        
        # Roll yield is negative of term structure (backwardation = positive carry)
        df['RollYield'] = -df['TermStructure_Smooth'] / 100  # Convert to decimal
        
        # Annualize if requested (assume monthly rebalancing)
        if self.annualize:
            df['Signal'] = df['RollYield'] * 12
        else:
            df['Signal'] = df['RollYield']
        
        return df


class CombinedCarry(SignalModel):
    """
    Combined Carry signal using multiple carry sources.
    
    Combines yield carry, roll yield, and currency carry into unified signal.
    Useful for multi-asset portfolios with bonds, commodities, and FX exposure.
    
    Args:
        weights: Dict of weights for each carry component (default: equal)
        min_components: Minimum number of components required (default: 1)
    
    Returns:
        DataFrame with 'Signal' column (weighted average of carry signals)
    
    Example:
        >>> carry = CombinedCarry(weights={'yield': 0.6, 'roll': 0.4})
        >>> # DataFrame should have carry components as columns
        >>> signals = carry.generate(data_with_yields_and_roll)
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_components: int = 1
    ):
        self.weights = weights or {'yield': 1.0}
        self.min_components = min_components
        
        # Validate weights sum to 1
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate combined carry signal.
        
        Args:
            df: DataFrame with carry component columns:
                - 'YieldCarry': Yield-based carry
                - 'RollYield': Futures roll yield
                - 'CurrencyCarry': FX carry (optional)
        
        Returns:
            DataFrame with 'Signal' column (weighted carry)
        """
        df = df.copy()
        
        # Map weight keys to column names
        component_map = {
            'yield': 'YieldCarry',
            'roll': 'RollYield',
            'currency': 'CurrencyCarry'
        }
        
        # Calculate weighted signal
        df['Signal'] = 0.0
        components_found = 0
        
        for component, weight in self.weights.items():
            col_name = component_map.get(component, component)
            
            if col_name in df.columns:
                df['Signal'] += df[col_name] * weight
                components_found += 1
        
        # Check minimum components
        if components_found < self.min_components:
            raise ValueError(
                f"Found {components_found} carry components, but min_components={self.min_components}"
            )
        
        return df
