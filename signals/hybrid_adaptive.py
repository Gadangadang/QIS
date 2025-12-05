import pandas as pd
import numpy as np
from signals.base import SignalModel


class HybridAdaptiveSignal(SignalModel):
    """
    Adaptive signal that switches between mean reversion and momentum based on volatility regime.
    
    - HIGH VOLATILITY: Mean reversion (buy dips, sell rips)
    - LOW VOLATILITY: Momentum (trend following)
    """
    
    def __init__(
        self, 
        vol_window: int = 50,
        vol_threshold: float = 0.012,  # Annualized vol threshold (~19% annual)
        mr_window: int = 20,         # Mean reversion lookback
        mr_entry_z: float = 1.5,       # Z-score to enter mean reversion
        mr_exit_z: float = 0.5,        # Z-score to exit mean reversion
        mom_fast: int = 20,          # Momentum fast MA
        mom_slow: int = 50,          # Momentum slow MA
    ):
        """
        Initialize hybrid adaptive signal with type validation.
        
        Args:
            vol_window: Volatility calculation window
            vol_threshold: Threshold for high volatility regime
            mr_window: Mean reversion window
            mr_entry_z: Z-score for mean reversion entry
            mr_exit_z: Z-score for mean reversion exit
            mom_fast: Fast moving average period
            mom_slow: Slow moving average period
        
        Raises:
            ValueError: If parameters are invalid
        """
        if vol_window < 2:
            raise ValueError(f"vol_window must be >= 2, got {vol_window}")
        if vol_threshold <= 0:
            raise ValueError(f"vol_threshold must be positive, got {vol_threshold}")
        if mr_window < 2:
            raise ValueError(f"mr_window must be >= 2, got {mr_window}")
        if mr_entry_z <= 0:
            raise ValueError(f"mr_entry_z must be positive, got {mr_entry_z}")
        if mr_exit_z < 0:
            raise ValueError(f"mr_exit_z must be non-negative, got {mr_exit_z}")
        if mr_exit_z >= mr_entry_z:
            raise ValueError(f"mr_exit_z ({mr_exit_z}) must be < mr_entry_z ({mr_entry_z})")
        if mom_fast < 1:
            raise ValueError(f"mom_fast must be >= 1, got {mom_fast}")
        if mom_slow <= mom_fast:
            raise ValueError(f"mom_slow ({mom_slow}) must be > mom_fast ({mom_fast})")
        
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.mr_window = mr_window
        self.mr_entry_z = mr_entry_z
        self.mr_exit_z = mr_exit_z
        self.mom_fast = mom_fast
        self.mom_slow = mom_slow

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate hybrid adaptive signals using vectorized operations.
        
        Switches between mean reversion (high vol) and momentum (low vol) regimes.
        
        Args:
            df: DataFrame with at least 'Close' column
        
        Returns:
            DataFrame with added columns:
                - Volatility: Rolling volatility of returns
                - HighVol: Boolean indicating high volatility regime
                - MR_Z: Z-score for mean reversion strategy
                - MA_Fast, MA_Slow: Moving averages for momentum
                - Signal: Trading signal (1=long, -1=short, 0=flat)
        
        Raises:
            ValueError: If df is empty or missing 'Close' column
        
        Logic:
            HIGH VOL REGIME (HighVol=1): Mean Reversion
            - Long when Z < -entry_z (oversold)
            - Short when Z > +entry_z (overbought)
            - Exit when Z reverts or hits stop loss
            
            LOW VOL REGIME (HighVol=0): Momentum
            - Long when price > MA_Fast > MA_Slow (uptrend)
            - Short when price < MA_Fast < MA_Slow (downtrend)
            - Exit when trend breaks
        
        Note:
            Uses vectorized operations with boolean masks.
            Warmup period: max(vol_window, mom_slow)
        """
        # Validate input
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must have 'Close' column")
        
        df = df.copy()
        close = df["Close"]
        
        # Calculate volatility regime (rolling std of returns)
        returns = close.pct_change()
        df["Volatility"] = returns.rolling(self.vol_window).std()
        df["HighVol"] = (df["Volatility"] > self.vol_threshold).astype(int)
        
        # Mean Reversion components
        mr_sma = close.rolling(self.mr_window).mean()
        mr_std = close.rolling(self.mr_window).std()
        df["MR_Z"] = (close - mr_sma) / mr_std
        
        # Momentum components
        df["MA_Fast"] = close.rolling(self.mom_fast).mean()
        df["MA_Slow"] = close.rolling(self.mom_slow).mean()
        
        # Initialize signal
        df['Signal'] = 0
        warmup = max(self.vol_window, self.mom_slow)
        
        # HIGH VOLATILITY REGIME: Mean Reversion (vectorized)
        high_vol_mask = df["HighVol"] == 1
        
        # MR Entry signals
        mr_long_entry = high_vol_mask & (df["MR_Z"] <= -self.mr_entry_z)
        mr_short_entry = high_vol_mask & (df["MR_Z"] >= self.mr_entry_z)
        
        # Apply MR entries
        df.loc[mr_long_entry, 'Signal'] = 1
        df.loc[mr_short_entry, 'Signal'] = -1
        
        # Forward fill MR positions
        df['Signal'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).astype(int)
        
        # MR Exit conditions (vectorized)
        mr_long_exit = high_vol_mask & (df['Signal'] == 1) & (
            (df["MR_Z"] >= -self.mr_exit_z) | (df["MR_Z"] > self.mr_entry_z)
        )
        mr_short_exit = high_vol_mask & (df['Signal'] == -1) & (
            (df["MR_Z"] <= self.mr_exit_z) | (df["MR_Z"] < -self.mr_entry_z)
        )
        
        # Apply MR exits
        df.loc[mr_long_exit, 'Signal'] = 0
        df.loc[mr_short_exit, 'Signal'] = 0
        
        # LOW VOLATILITY REGIME: Momentum (vectorized)
        low_vol_mask = df["HighVol"] == 0
        
        # Momentum Entry signals
        mom_long_entry = low_vol_mask & (close > df["MA_Fast"]) & (df["MA_Fast"] > df["MA_Slow"])
        mom_short_entry = low_vol_mask & (close < df["MA_Fast"]) & (df["MA_Fast"] < df["MA_Slow"])
        
        # Apply momentum entries
        df.loc[mom_long_entry, 'Signal'] = 1
        df.loc[mom_short_entry, 'Signal'] = -1
        
        # Forward fill momentum positions
        df['Signal'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).astype(int)
        
        # Momentum Exit conditions (vectorized)
        mom_long_exit = low_vol_mask & (df['Signal'] == 1) & (close < df["MA_Fast"])
        mom_short_exit = low_vol_mask & (df['Signal'] == -1) & (close > df["MA_Fast"])
        
        # Apply momentum exits
        df.loc[mom_long_exit, 'Signal'] = 0
        df.loc[mom_short_exit, 'Signal'] = 0
        
        # Final forward fill and cleanup
        df['Signal'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).astype(int)
        
        # Clear warmup period
        df.iloc[:warmup, df.columns.get_loc('Signal')] = 0

        df['Signal'] = df['Signal'].astype(int)
        return df
