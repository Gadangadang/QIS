"""Ensemble signal strategies combining multiple signal generators."""
import pandas as pd
import numpy as np
from signals.base import SignalModel
from signals.mean_reversion import MeanReversionSignal
from signals.momentum import MomentumSignal


class EnsembleSignal(SignalModel):
    """
    Ensemble signal combining multiple momentum strategies with trend filter.
    
    Averages positions from multiple momentum signals with different lookback
    periods, then applies a trend filter (SMA) to avoid bear markets.
    
    Strategy:
        1. Generate signals from 3 momentum models (20, 50, 100-day lookback)
        2. Average the positions (majority vote)
        3. Apply trend filter: only trade when above 50-day SMA
    
    This approach reduces whipsaws by requiring consensus among different
    timeframe momentum signals.
    
    Example:
        >>> signal = EnsembleSignal()
        >>> df_with_signals = signal.generate(price_data)
    """
    
    def __init__(self):
        """
        Initialize ensemble with predefined momentum strategies.
        
        Uses 3 momentum signals:
        - Fast: 20-day lookback, 2.5% threshold
        - Medium: 50-day lookback, 2.0% threshold  
        - Slow: 100-day lookback, 1.8% threshold
        
        Note:
            Mean reversion signals are commented out but can be added
            for a mixed trend/counter-trend ensemble.
        """
        self.signals = [
            MomentumSignal(lookback=20,  threshold=0.025),
            MomentumSignal(lookback=50, threshold=0.02),
            MomentumSignal(lookback=100, threshold=0.018),
            #MeanReversionSignal(window=8, entry_z=2.3, exit_z=0.8),
            #MeanReversionSignal(window=15, entry_z=2.1, exit_z=0.8),
            #MeanReversionSignal(window=40, entry_z=1.9, exit_z=0.8),
        ]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble signals by averaging component strategies.
        
        Args:
            df (pd.DataFrame): DataFrame with at least 'Close' column
        
        Returns:
            pd.DataFrame: Original DataFrame with added columns:
                - EnsemblePosition: Average of all component signals
                - TrendFilter: 1 if above 50-day SMA, 0 otherwise
                - Position: Final signal (EnsemblePosition * TrendFilter)
        
        Logic:
            1. Generate signals from all component strategies
            2. Average positions and round (majority vote)
            3. Apply trend filter (only long above 50-day SMA)
            4. Final position is 1 (long) or 0 (flat), never short
        """
        
        df = df.copy()
        positions = [sig.generate(df.copy())["Position"] for sig in self.signals]
        pos_df = pd.concat(positions, axis=1)
        df["EnsemblePosition"] = pos_df.mean(axis=1).round().clip(-1, 1).astype(int)
        # Only go long when above 200-day MA. Never short. Ever.
        df["TrendFilter"] = (df["Close"] > df["Close"].rolling(50).mean()).astype(int)
        df["Position"] = (
            df["EnsemblePosition"] * df["TrendFilter"]
            )  # zero out when below

        return df


class EnsembleSignalNew(SignalModel):
    """
    Advanced ensemble using multiple mean reversion signals with bull market filter.
    
    Strategy:
        1. Only trade during bull markets (price > 200-day SMA)
        2. Use 3 mean reversion signals (10, 20, 60-day windows)
        3. Combine via majority vote
        4. Final position only active in bull market
    
    This approach captures mean reversion opportunities while avoiding
    getting chopped up in bear markets.
    
    Example:
        >>> signal = EnsembleSignalNew()
        >>> df_with_signals = signal.generate(price_data)
    """
    
    def __init__(self):
        """Initialize ensemble with no parameters (fixed configuration)."""
        pass

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble mean reversion signals with trend filter.
        
        Args:
            df (pd.DataFrame): DataFrame with at least 'Close' column
        
        Returns:
            pd.DataFrame: Original DataFrame with added columns:
                - SMA200: 200-day simple moving average
                - InBullMarket: Boolean, True when price > SMA200
                - MR_Sum: Sum of mean reversion signals
                - MR_Vote: Majority vote from MR signals (-1, 0, +1)
                - Position: Final signal (MR_Vote when in bull market, else 0)
        
        Component Signals:
            - MR10: 10-day window, ±2.2σ entry, ±1.0σ exit
            - MR20: 20-day window, ±2.0σ entry, ±1.0σ exit
            - MR60: 60-day window, ±1.8σ entry, ±1.0σ exit
        
        Note:
            First 200 bars set to Position = 0 for burn-in period.
        """
        df = df.copy()

        # 1. Trend filter — the only thing that matters long-term
        df["SMA200"] = df["Close"].rolling(200).mean()
        df["InBullMarket"] = df["Close"] > df["SMA200"]

        # 2. Only use mean-reversion signals (they are fast and high-Sharpe)
        mr10 = MeanReversionSignal(window=10, entry_z=2.2, exit_z=1.0).generate(
            df.copy()
        )
        mr20 = MeanReversionSignal(window=20, entry_z=2.0, exit_z=1.0).generate(
            df.copy()
        )
        mr60 = MeanReversionSignal(window=60, entry_z=1.8, exit_z=1.0).generate(
            df.copy()
        )

        # 3. Combine MR signals (majority vote)
        df["MR_Sum"] = mr10["Position"] + mr20["Position"] + mr60["Position"]
        df["MR_Vote"] = np.sign(df["MR_Sum"])  # -1, 0, or +1

        # 4. FINAL POSITION: only trade MR when in bull market
        df["Position"] = df["MR_Vote"].where(df["InBullMarket"], 0)

        # 5. Burn-in
        df.iloc[:200, df.columns.get_loc("Position")] = 0

        return df
