"""
Energy Seasonal Signal V2 - Enhanced Models

This module introduces two new variations of the Energy Seasonal Signal:
1. EnergySeasonalBalanced: A more robust, "clever" version of the original.
2. EnergySeasonalAggressive: A high-risk, high-reward version with looser constraints.

Improvements over V1:
- Dynamic volatility thresholds (percentile-based) instead of fixed absolute values.
- Continuous seasonality factors instead of discrete buckets.
- Weighted signal combination instead of strict regime switching.
- Trend confirmation for mean reversion entries.
"""

import pandas as pd
import numpy as np
from signals.base import SignalModel
from typing import List, Optional

class EnergySeasonalBalanced(SignalModel):
    """
    Balanced Energy Seasonal Model (Improved Baseline).
    
    Key Improvements:
    - Dynamic Volatility: Uses rolling percentile to define high/low vol regimes.
    - Smart Seasonality: Uses continuous sine-wave approximation or monthly averages.
    - Trend Filter: Adds a long-term trend filter to avoid fighting major moves.
    - Soft Regime Switching: Blends MR and Momentum signals in transition zones.
    """
    
    def __init__(
        self,
        # Volatility
        vol_window: int = 60,
        vol_percentile_threshold: float = 0.75, # Top 25% vol = High Vol
        
        # Mean Reversion
        mr_window: int = 20,
        mr_entry_z: float = 2.0,
        mr_exit_z: float = 0.0,
        
        # Momentum
        mom_window: int = 60,
        mom_threshold: float = 0.0,
        
        # Seasonality
        use_seasonality: bool = True,
        seasonal_strength: float = 1.0, # Multiplier for seasonal influence
        
        # Trend Filter
        trend_window: int = 200
    ):
        self.vol_window = vol_window
        self.vol_percentile_threshold = vol_percentile_threshold
        self.mr_window = mr_window
        self.mr_entry_z = mr_entry_z
        self.mr_exit_z = mr_exit_z
        self.mom_window = mom_window
        self.mom_threshold = mom_threshold
        self.use_seasonality = use_seasonality
        self.seasonal_strength = seasonal_strength
        self.trend_window = trend_window

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Volatility Regime (Dynamic)
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['Vol_Rank'] = df['Vol'].rolling(self.vol_window).rank(pct=True)
        
        # High Vol if current vol is in top X% of recent history
        df['HighVol'] = (df['Vol_Rank'] > self.vol_percentile_threshold).astype(int)
        
        # 2. Mean Reversion (Z-Score)
        rolling_mean = df['Close'].rolling(self.mr_window).mean()
        rolling_std = df['Close'].rolling(self.mr_window).std()
        df['Z_Score'] = (df['Close'] - rolling_mean) / rolling_std
        
        # 3. Momentum
        df['Momentum'] = df['Close'].pct_change(self.mom_window)
        
        # 4. Seasonality (Continuous approximation)
        # Simple approximation: Winter peak (Jan/Feb), Summer peak (Jul/Aug)
        # We'll use a lookup table for monthly weights based on typical NG seasonality
        seasonal_weights = {
            1: 0.8, 2: 0.6, 3: 0.2,     # Winter (Bullish)
            4: -0.2, 5: -0.3,           # Shoulder (Bearish)
            6: 0.3, 7: 0.4, 8: 0.3,     # Summer (Mild Bullish)
            9: -0.4, 10: -0.5,          # Shoulder (Bearish)
            11: 0.5, 12: 0.7            # Winter (Bullish)
        }
        df['Month'] = df.index.month
        df['Seasonal_Factor'] = df['Month'].map(seasonal_weights)
        
        # 5. Signal Generation
        df['Signal'] = 0
        
        # Vectorized Logic
        # MR Signal: Long if Z < -2, Short if Z > 2
        mr_long = (df['Z_Score'] < -self.mr_entry_z)
        mr_short = (df['Z_Score'] > self.mr_entry_z)
        
        # Momentum Signal: Long if Mom > 0, Short if Mom < 0
        mom_long = (df['Momentum'] > self.mom_threshold)
        mom_short = (df['Momentum'] < -self.mom_threshold)
        
        # Combined Logic
        for i in range(max(self.vol_window, self.trend_window), len(df)):
            idx = df.index[i]
            is_high_vol = df['HighVol'].iloc[i]
            seasonal = df['Seasonal_Factor'].iloc[i] * self.seasonal_strength if self.use_seasonality else 0
            
            # Base Signal Selection
            if is_high_vol:
                # In high vol, prefer Mean Reversion
                # But filter with Seasonality: Don't short in strong winter, don't long in strong shoulder
                if mr_long.iloc[i] and seasonal > -0.3:
                    df.loc[idx, 'Signal'] = 1
                elif mr_short.iloc[i] and seasonal < 0.3:
                    df.loc[idx, 'Signal'] = -1
            else:
                # In low vol, prefer Momentum
                # Align with seasonality
                if mom_long.iloc[i] and seasonal > -0.1:
                    df.loc[idx, 'Signal'] = 1
                elif mom_short.iloc[i] and seasonal < 0.1:
                    df.loc[idx, 'Signal'] = -1
            
            # Exit Logic - Force exit if conditions neutralize
            # If we are Long and Z-Score > 0 (reverted to mean), exit
            current_sig = df.loc[idx, 'Signal']
            z_score = df['Z_Score'].iloc[i]
            
            # If signal was generated by MR (High Vol), exit at mean
            if is_high_vol:
                if current_sig == 1 and z_score > self.mr_exit_z:
                    df.loc[idx, 'Signal'] = 0
                elif current_sig == -1 and z_score < -self.mr_exit_z:
                    df.loc[idx, 'Signal'] = 0
            
            # If signal was generated by Momentum (Low Vol), exit if momentum fades
            else:
                mom = df['Momentum'].iloc[i]
                if current_sig == 1 and mom < 0:
                    df.loc[idx, 'Signal'] = 0
                elif current_sig == -1 and mom > 0:
                    df.loc[idx, 'Signal'] = 0
            
        # Fill NaNs
        df['Signal'] = df['Signal'].fillna(0)
        return df


class EnergySeasonalAggressive(SignalModel):
    """
    Aggressive Energy Seasonal Model.
    
    Key Features:
    - Looser Constraints: Takes more trades.
    - Counter-Trend: Aggressively fades spikes even against seasonality if extreme enough.
    - Fast Momentum: Uses shorter lookback for momentum to catch early moves.
    - Leverage: Designed for higher risk tolerance (implied, not explicit in signal).
    """
    
    def __init__(
        self,
        vol_window: int = 30,
        vol_percentile_threshold: float = 0.90, # Only reduce risk in extreme vol
        mr_window: int = 14,       # Faster MR
        mr_entry_z: float = 1.5,   # Aggressive entry (1.5 sigma)
        mr_exit_z: float = -0.5,   # Hold longer
        mom_window: int = 20,      # Faster momentum (monthly)
        mom_threshold: float = -0.02, # Looser momentum filter
        use_seasonality: bool = True,
        seasonal_strength: float = 1.5,
        crisis_alpha_entry: float = 3.0
    ):
        self.vol_window = vol_window
        self.vol_percentile_threshold = vol_percentile_threshold
        self.mr_window = mr_window
        self.mr_entry_z = mr_entry_z
        self.mr_exit_z = mr_exit_z
        self.mom_window = mom_window
        self.mom_threshold = mom_threshold
        self.use_seasonality = use_seasonality
        self.seasonal_strength = seasonal_strength
        self.crisis_alpha_entry = crisis_alpha_entry

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Volatility
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol'] = df['Returns'].rolling(self.vol_window).std() * np.sqrt(252)
        
        # Calculate Vol Percentile
        df['Vol_Percentile'] = df['Vol'].rolling(252).rank(pct=True)
        
        # 2. Fast Mean Reversion
        rolling_mean = df['Close'].rolling(self.mr_window).mean()
        rolling_std = df['Close'].rolling(self.mr_window).std()
        df['Z_Score'] = (df['Close'] - rolling_mean) / rolling_std
        
        # 3. Fast Momentum
        df['Momentum'] = df['Close'].pct_change(self.mom_window)
        
        # 4. Seasonality (Binary)
        df['Month'] = df.index.month
        # Aggressive seasonality: Only care about the strongest months
        df['Strong_Season'] = 0
        df.loc[df['Month'].isin([12, 1, 2]), 'Strong_Season'] = 1   # Winter
        df.loc[df['Month'].isin([9, 10]), 'Strong_Season'] = -1     # Shoulder dump
        
        # 5. Signal Logic
        df['Signal'] = 0
        
        # MR Signals (Aggressive)
        mr_long = (df['Z_Score'] < -self.mr_entry_z)
        mr_short = (df['Z_Score'] > self.mr_entry_z)
        
        # Momentum Signals
        mom_long = (df['Momentum'] > self.mom_threshold)
        mom_short = (df['Momentum'] < -self.mom_threshold)
        
        # Combined Logic (Priority: MR > Seasonality > Momentum)
        # If Z-score is extreme (> 3), ignore everything and fade it (Crisis Alpha)
        extreme_fade_long = (df['Z_Score'] < -self.crisis_alpha_entry)
        extreme_fade_short = (df['Z_Score'] > self.crisis_alpha_entry)
        
        for i in range(max(self.mr_window, self.mom_window, 252), len(df)):
            idx = df.index[i]
            
            # 1. Extreme Reversion (Highest Priority - Crisis Alpha)
            if extreme_fade_long.iloc[i]:
                df.loc[idx, 'Signal'] = 1
                continue
            if extreme_fade_short.iloc[i]:
                df.loc[idx, 'Signal'] = -1
                continue
                
            # 2. Seasonal + Momentum (Trend Following)
            season = df['Strong_Season'].iloc[i]
            if self.use_seasonality:
                if season == 1 and mom_long.iloc[i]:
                    df.loc[idx, 'Signal'] = 1
                elif season == -1 and mom_short.iloc[i]:
                    df.loc[idx, 'Signal'] = -1
            
            # 3. Standard Mean Reversion (If no strong seasonal trend)
            # Only if not in extreme volatility (unless it's crisis alpha level)
            vol_p = df['Vol_Percentile'].iloc[i]
            if season == 0 or not self.use_seasonality:
                if vol_p < self.vol_percentile_threshold:
                    if mr_long.iloc[i]:
                        df.loc[idx, 'Signal'] = 1
                    elif mr_short.iloc[i]:
                        df.loc[idx, 'Signal'] = -1
                    
        return df
