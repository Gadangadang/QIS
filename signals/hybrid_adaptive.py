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
        vol_window=50,
        vol_threshold=0.012,  # Annualized vol threshold (~19% annual)
        mr_window=20,         # Mean reversion lookback
        mr_entry_z=1.5,       # Z-score to enter mean reversion
        mr_exit_z=0.5,        # Z-score to exit mean reversion
        mom_fast=20,          # Momentum fast MA
        mom_slow=50,          # Momentum slow MA
    ):
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.mr_window = mr_window
        self.mr_entry_z = mr_entry_z
        self.mr_exit_z = mr_exit_z
        self.mom_fast = mom_fast
        self.mom_slow = mom_slow

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
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
        
        # Initialize position
        df['Signal'] = 0
        
        # Build positions bar-by-bar
        warmup = max(self.vol_window, self.mom_slow)
        
        for i in range(warmup, len(df)):
            prev_position = df.iloc[i - 1]['Signal']
            high_vol = df.iloc[i]["HighVol"]
            z_score = df.iloc[i]["MR_Z"]
            price = df.iloc[i]["Close"]
            ma_fast = df.iloc[i]["MA_Fast"]
            ma_slow = df.iloc[i]["MA_Slow"]
            
            # HIGH VOLATILITY REGIME: Mean Reversion
            if high_vol == 1:
                if prev_position == 0:
                    # Enter on extreme deviations
                    if z_score <= -self.mr_entry_z:
                        df.iloc[i, df.columns.get_loc('Signal')] = 1  # Long
                    elif z_score >= self.mr_entry_z:
                        df.iloc[i, df.columns.get_loc('Signal')] = -1  # Short
                    else:
                        df.iloc[i, df.columns.get_loc('Signal')] = 0
                        
                elif prev_position == 1:
                    # Exit long when mean reverted
                    if z_score >= -self.mr_exit_z or z_score > self.mr_entry_z:
                        df.iloc[i, df.columns.get_loc('Signal')] = 0
                    else:
                        df.iloc[i, df.columns.get_loc('Signal')] = 1
                        
                elif prev_position == -1:
                    # Exit short when mean reverted
                    if z_score <= self.mr_exit_z or z_score < -self.mr_entry_z:
                        df.iloc[i, df.columns.get_loc('Signal')] = 0
                    else:
                        df.iloc[i, df.columns.get_loc('Signal')] = -1
            
            # LOW VOLATILITY REGIME: Momentum
            else:
                if prev_position == 0:
                    # Enter on trend alignment
                    if price > ma_fast and ma_fast > ma_slow:
                        df.iloc[i, df.columns.get_loc('Signal')] = 1  # Long trend
                    elif price < ma_fast and ma_fast < ma_slow:
                        df.iloc[i, df.columns.get_loc('Signal')] = -1  # Short trend
                    else:
                        df.iloc[i, df.columns.get_loc('Signal')] = 0
                        
                elif prev_position == 1:
                    # Exit long if trend breaks
                    if price < ma_fast:
                        df.iloc[i, df.columns.get_loc('Signal')] = 0
                    else:
                        df.iloc[i, df.columns.get_loc('Signal')] = 1
                        
                elif prev_position == -1:
                    # Exit short if trend breaks
                    if price > ma_fast:
                        df.iloc[i, df.columns.get_loc('Signal')] = 0
                    else:
                        df.iloc[i, df.columns.get_loc('Signal')] = -1

        df['Signal'] = df['Signal'].astype(int)
        return df
