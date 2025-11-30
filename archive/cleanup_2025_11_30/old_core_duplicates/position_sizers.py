"""Position sizing methods for dynamic allocation."""
import pandas as pd
import numpy as np
from typing import Optional


class VolatilityTargeting:
    """Scale positions to achieve target volatility.
    
    This transforms binary positions {-1, 0, 1} into dynamic sizing
    by scaling inversely with realized volatility:
        size = target_vol / realized_vol
    
    Example: If target is 12% annual vol and current realized vol is 24%,
             position size = 12% / 24% = 0.5 (half position)
    """
    
    def __init__(
        self,
        target_vol: float = 0.12,
        lookback: int = 20,
        min_periods: int = 10,
        max_leverage: float = 2.0,
        min_size: float = 0.0,
        annualization_factor: float = 252,
    ):
        """Initialize volatility targeting.
        
        Args:
            target_vol: Target annualized volatility (e.g., 0.12 = 12%)
            lookback: Rolling window for volatility calculation (days)
            min_periods: Minimum periods required for vol calculation
            max_leverage: Maximum position size multiplier (e.g., 2.0 = 200%)
            min_size: Minimum position size (0.0 = can go to zero)
            annualization_factor: Days per year for vol annualization (252 for daily)
        """
        self.target_vol = target_vol
        self.lookback = lookback
        self.min_periods = min_periods
        self.max_leverage = max_leverage
        self.min_size = min_size
        self.annualization_factor = annualization_factor
    
    def calculate_size(self, df: pd.DataFrame, position_col: str = "Position") -> pd.Series:
        """Calculate position sizes based on volatility targeting.
        
        Args:
            df: DataFrame with price data and Position column
            position_col: Column containing raw signals {-1, 0, 1}
            
        Returns:
            Series with sized positions (direction * size)
        """
        if position_col not in df.columns:
            raise ValueError(f"Position column '{position_col}' not found")
        
        if "Close" not in df.columns:
            raise ValueError("Close price column required for volatility calculation")
        
        # Calculate returns
        returns = df["Close"].pct_change()
        
        # Calculate realized volatility (annualized)
        realized_vol = returns.rolling(
            window=self.lookback,
            min_periods=self.min_periods
        ).std() * np.sqrt(self.annualization_factor)
        
        # Calculate size multiplier: target_vol / realized_vol
        size_multiplier = self.target_vol / realized_vol
        
        # Apply leverage caps
        size_multiplier = size_multiplier.clip(lower=self.min_size, upper=self.max_leverage)
        
        # Fill NaN with 1.0 (no sizing adjustment when vol can't be calculated)
        size_multiplier = size_multiplier.fillna(1.0)
        
        # Apply sizing to raw position signal
        sized_position = df[position_col] * size_multiplier
        
        return sized_position


class KellyCriterion:
    """Kelly criterion position sizing based on historical win rate and payoff.
    
    Kelly formula: f* = (p*b - q) / b
    where:
        p = win rate
        q = 1 - p (loss rate)
        b = avg_win / avg_loss (payoff ratio)
        f* = optimal fraction of capital to bet
    
    Uses fractional Kelly (typically 25-50%) to reduce risk.
    """
    
    def __init__(
        self,
        lookback: int = 50,
        min_trades: int = 10,
        kelly_fraction: float = 0.25,
        max_size: float = 1.0,
        min_size: float = 0.0,
    ):
        """Initialize Kelly criterion sizing.
        
        Args:
            lookback: Number of recent trades to use for win/loss stats
            min_trades: Minimum trades required before applying Kelly
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
            max_size: Maximum position size
            min_size: Minimum position size
        """
        self.lookback = lookback
        self.min_trades = min_trades
        self.kelly_fraction = kelly_fraction
        self.max_size = max_size
        self.min_size = min_size
    
    def calculate_size(
        self,
        df: pd.DataFrame,
        position_col: str = "Position",
        trade_history: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """Calculate position sizes based on Kelly criterion.
        
        Note: This requires trade history to calculate win rate and payoffs.
        Without trade history, returns fixed sizing (1.0).
        
        Args:
            df: DataFrame with price data and Position column
            position_col: Column containing raw signals
            trade_history: DataFrame with completed trades (columns: PnL_pct)
            
        Returns:
            Series with sized positions
        """
        if trade_history is None or len(trade_history) < self.min_trades:
            # Not enough trade history - use fixed sizing
            return df[position_col].copy()
        
        # Use only recent trades
        recent_trades = trade_history.tail(self.lookback)
        
        # Calculate win rate
        wins = recent_trades[recent_trades["PnL_pct"] > 0]
        losses = recent_trades[recent_trades["PnL_pct"] <= 0]
        
        if len(wins) == 0 or len(losses) == 0:
            # Can't calculate Kelly without both wins and losses
            return df[position_col].copy()
        
        p = len(wins) / len(recent_trades)  # win rate
        q = 1 - p  # loss rate
        
        avg_win = wins["PnL_pct"].mean()
        avg_loss = abs(losses["PnL_pct"].mean())
        
        if avg_loss == 0:
            return df[position_col].copy()
        
        b = avg_win / avg_loss  # payoff ratio
        
        # Kelly formula: f* = (p*b - q) / b
        kelly_size = (p * b - q) / b
        
        # Apply fractional Kelly
        kelly_size = kelly_size * self.kelly_fraction
        
        # Clip to min/max
        kelly_size = np.clip(kelly_size, self.min_size, self.max_size)
        
        # Apply to all positions (constant sizing across time period)
        sized_position = df[position_col] * kelly_size
        
        return sized_position


class SignalStrengthSizing:
    """Scale position size based on signal strength.
    
    Stronger signals get larger positions. For example:
    - Momentum: larger positions when momentum is very positive/negative
    - Mean reversion: larger positions when Z-score is further from mean
    """
    
    def __init__(
        self,
        strength_col: str = "SignalStrength",
        base_size: float = 0.5,
        max_size: float = 1.5,
        min_size: float = 0.0,
    ):
        """Initialize signal strength sizing.
        
        Args:
            strength_col: Column containing signal strength (absolute value used)
            base_size: Base position size for average signal
            max_size: Maximum position size
            min_size: Minimum position size (when signal strength is 0)
        """
        self.strength_col = strength_col
        self.base_size = base_size
        self.max_size = max_size
        self.min_size = min_size
    
    def calculate_size(self, df: pd.DataFrame, position_col: str = "Position") -> pd.Series:
        """Calculate position sizes based on signal strength.
        
        Args:
            df: DataFrame with Position and SignalStrength columns
            position_col: Column containing raw signals {-1, 0, 1}
            
        Returns:
            Series with sized positions
        """
        if self.strength_col not in df.columns:
            raise ValueError(f"Signal strength column '{self.strength_col}' not found")
        
        # Get signal direction and strength
        direction = df[position_col]
        strength = df[self.strength_col].abs()
        
        # Normalize strength to [0, 1] range
        if strength.max() > 0:
            strength_normalized = strength / strength.max()
        else:
            strength_normalized = pd.Series(0, index=df.index)
        
        # Scale size: min_size + (max_size - min_size) * strength
        size = self.min_size + (self.max_size - self.min_size) * strength_normalized
        
        # Apply sizing to direction
        sized_position = direction * size
        
        return sized_position


class FixedSizing:
    """Fixed position sizing (baseline/control).
    
    Simply returns the raw signal positions without modification.
    Useful as a baseline for comparing against dynamic sizing methods.
    """
    
    def __init__(self, size: float = 1.0):
        """Initialize fixed sizing.
        
        Args:
            size: Fixed size multiplier for all positions
        """
        self.size = size
    
    def calculate_size(self, df: pd.DataFrame, position_col: str = "Position") -> pd.Series:
        """Return fixed-sized positions.
        
        Args:
            df: DataFrame with Position column
            position_col: Column containing raw signals
            
        Returns:
            Series with fixed-sized positions
        """
        return df[position_col] * self.size
