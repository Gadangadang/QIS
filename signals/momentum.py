"""Momentum-based trading signals for trend-following strategies."""
import pandas as pd
import numpy as np
from signals.base import SignalModel


class MomentumSignal(SignalModel):
    """
    Basic momentum signal generator using price rate of change.
    
    Generates long signals when momentum exceeds threshold and exits when
    momentum reverses below exit threshold. Uses forward-fill to maintain
    positions between entry and exit signals.
    
    Attributes:
        lookback (int): Number of periods to calculate momentum
        threshold (float): Minimum momentum for entry (e.g., 0.02 = 2%)
        exit_threshold (float): Momentum level for exit (e.g., 0.0 = flat)
    """
    
    def __init__(self, lookback=20, threshold=0.02, exit_threshold=0.0):
        """
        Initialize momentum signal generator.
        
        Args:
            lookback (int): Lookback period for momentum calculation. Default 20.
            threshold (float): Entry threshold as decimal (0.02 = 2% gain required). Default 0.02.
            exit_threshold (float): Exit threshold as decimal (0.0 = no gain). Default 0.0.
        """
        self.lookback = lookback
        self.threshold = threshold
        self.exit_threshold = exit_threshold
        

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum signals for given price data.
        
        Args:
            df (pd.DataFrame): DataFrame with at least 'Close' column
        
        Returns:
            pd.DataFrame: Original DataFrame with added columns:
                - Momentum: Rate of change over lookback period
                - Position: Trading signal (1=long, 0=flat)
        
        Note:
            First (lookback + 20) bars are set to 0 for burn-in period.
        """
        df = df.copy()
        close = df["Close"]

        df["Momentum"] = close / close.shift(self.lookback) - 1

        # ONLY trade when we have valid momentum AND it's been valid for a while
        df["Position"] = 0
        # Only go long when strong AND positive
        enter = (df["Momentum"] > self.threshold) & (df["Momentum"] > 0)
        exit = df["Momentum"] <= -self.exit_threshold

        # Apply entry
        df.loc[enter, "Position"] = 1
        # Apply exit (override)
        df.loc[exit, "Position"] = 0

        # Forward fill — but exit always wins
        df["Position"] = df["Position"].replace(0, np.nan).ffill(limit=None)
        df.loc[exit, "Position"] = 0  # FINAL OVERRIDE

        # Burn-in
        df.iloc[: self.lookback + 20, df.columns.get_loc("Position")] = 0

        df["Position"] = df["Position"].fillna(0).astype(int)

        return df


class MomentumSignalV2(SignalModel):
    """
    Enhanced momentum signal with trend filter (SMA) to avoid whipsaws.
    
    Only takes long positions when:
    1. Momentum exceeds entry threshold (strong trend)
    2. Price is above long-term SMA (bull market regime)
    
    Exits when either:
    1. Momentum falls below exit threshold (trend weakening)
    2. Price crosses below SMA (regime change to bear)
    
    This is the recommended momentum signal for most applications as it
    significantly reduces false signals in ranging/bear markets.
    
    Attributes:
        lookback (int): Momentum calculation period
        entry_threshold (float): Minimum momentum for long entry
        exit_threshold (float): Momentum level triggering exit (typically negative)
        sma_filter (int): SMA period for regime filter
    
    Example:
        >>> signal = MomentumSignalV2(lookback=120, entry_threshold=0.02, 
        ...                           exit_threshold=-0.01, sma_filter=100)
        >>> df_with_signals = signal.generate(price_data)
    """
    
    def __init__(self, lookback=120, entry_threshold=0.02, exit_threshold=-0.01, sma_filter=100):
        """
        Initialize momentum signal with trend filter.
        
        Args:
            lookback (int): Lookback period for momentum calculation. Default 120 days.
            entry_threshold (float): Entry threshold (0.02 = 2% gain required). Default 0.02.
            exit_threshold (float): Exit threshold (negative = loss tolerance). Default -0.01.
            sma_filter (int): SMA period for trend filter. Default 100 days.
        
        Note:
            Typical settings:
            - Aggressive: lookback=60, entry=0.01, exit=-0.02
            - Conservative: lookback=180, entry=0.03, exit=-0.005
        """
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.sma_filter = sma_filter
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum signals with trend filter.
        
        Args:
            df (pd.DataFrame): DataFrame with at least 'Close' column
        
        Returns:
            pd.DataFrame: Original DataFrame with added columns:
                - Momentum: Rate of change over lookback period
                - SMA{n}: Simple moving average for trend filter
                - BullMarket: Boolean, True when price > SMA
                - Signal: Trading signal (1=long, 0=flat)
        
        Note:
            Warm-up period is max(lookback, sma_filter) + 20 bars.
            All positions during warm-up are set to 0.
        """
        df = df.copy()
        close = df["Close"]
        
        # Calculate momentum
        df["Momentum"] = close / close.shift(self.lookback) - 1
        
        # Regime filter (CRITICAL!)
        df[f"SMA{self.sma_filter}"] = close.rolling(self.sma_filter).mean()
        df["BullMarket"] = close > df[f"SMA{self.sma_filter}"]
        
        # Entry: Strong positive momentum in bull market
        enter = (df["Momentum"] > self.entry_threshold) & df["BullMarket"]
        
        # Exit: Either strong negative momentum OR bear market
        exit = (df["Momentum"] < self.exit_threshold) | ~df["BullMarket"]
        
        # Generate positions
        df["Signal"] = 0
        df.loc[enter, "Signal"] = 1
        df.loc[exit, "Signal"] = 0
        
        # Forward fill (stay in position until exit trigger)
        df["Signal"] = df["Signal"].replace(0, np.nan).ffill().fillna(0)
        
        # Burn-in (need max of lookback or sma_filter)
        warmup = max(self.lookback, self.sma_filter) + 20
        df.iloc[:warmup, df.columns.get_loc("Signal")] = 0
        
        return df