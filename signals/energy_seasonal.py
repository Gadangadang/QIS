"""
Energy Seasonal Signal - Combines Mean Reversion, Momentum, and Seasonality

Tailored for natural gas, heating oil, and other energy commodities with strong seasonal patterns.
"""
import pandas as pd
import numpy as np
from signals.base import SignalModel


class EnergySeasonalSignal(SignalModel):
    """
    Hybrid signal for energy commodities combining:
    1. Mean reversion (exploits volatility extremes)
    2. Momentum (captures sustained trends)
    3. Seasonality filters (aligns with natural gas demand cycles)
    
    Natural Gas Seasonality:
    - Winter (Nov-Mar): Heating demand → Bullish bias
    - Summer (Jun-Aug): Cooling demand (power gen) → Moderately bullish
    - Shoulder seasons (Apr-May, Sep-Oct): Lower demand → Neutral/bearish bias
    
    Signal Logic:
    - Uses hybrid approach: mean reversion in high volatility + momentum in low volatility
    - Applies seasonal filters to enhance/dampen signals based on time of year
    - Only takes trades aligned with seasonal expectations
    """
    
    def __init__(
        self,
        # Volatility regime detection
        vol_window: int = 30,
        vol_threshold: float = 0.015,  # ~24% annualized (high for commodities)
        
        # Mean reversion parameters (high vol regime)
        mr_window: int = 20,
        mr_entry_z: float = 2.0,    # Enter on 2σ extremes
        mr_exit_z: float = 0.5,     # Exit when returning to normal
        
        # Momentum parameters (low vol regime)
        mom_lookback: int = 60,     # Quarterly trend
        mom_threshold: float = 0.02, # 2% threshold to filter noise
        
        # Seasonality parameters
        use_seasonality: bool = True,
        winter_months: list = [11, 12, 1, 2, 3],  # Nov-Mar: heating season
        summer_months: list = [6, 7, 8],           # Jun-Aug: cooling season
        shoulder_months: list = [4, 5, 9, 10],     # Apr-May, Sep-Oct: low demand
        
        # Seasonal biases (-1 to +1, where +1 is bullish, -1 is bearish)
        winter_bias: float = 0.6,      # Strong bullish (heating demand)
        summer_bias: float = 0.3,      # Moderate bullish (power gen)
        shoulder_bias: float = -0.2,   # Slight bearish (low demand)
    ):
        """
        Initialize energy seasonal signal.
        
        Args:
            vol_window: Window for volatility regime detection
            vol_threshold: Daily vol threshold for high-vol regime
            mr_window: Lookback for mean reversion z-score
            mr_entry_z: Z-score threshold to enter mean reversion trade
            mr_exit_z: Z-score threshold to exit mean reversion trade
            mom_lookback: Lookback period for momentum calculation
            mom_threshold: Minimum return threshold for momentum signal
            use_seasonality: Whether to apply seasonal filters
            winter_months: List of winter month numbers (1=Jan)
            summer_months: List of summer month numbers
            shoulder_months: List of shoulder season month numbers
            winter_bias: Seasonal bias for winter (-1 to +1)
            summer_bias: Seasonal bias for summer (-1 to +1)
            shoulder_bias: Seasonal bias for shoulder seasons (-1 to +1)
        """
        # Validation
        if vol_window < 2:
            raise ValueError(f"vol_window must be >= 2, got {vol_window}")
        if mr_window < 2:
            raise ValueError(f"mr_window must be >= 2, got {mr_window}")
        if mom_lookback < 2:
            raise ValueError(f"mom_lookback must be >= 2, got {mom_lookback}")
        if not (-1 <= winter_bias <= 1):
            raise ValueError(f"winter_bias must be between -1 and 1, got {winter_bias}")
        if not (-1 <= summer_bias <= 1):
            raise ValueError(f"summer_bias must be between -1 and 1, got {summer_bias}")
        if not (-1 <= shoulder_bias <= 1):
            raise ValueError(f"shoulder_bias must be between -1 and 1, got {shoulder_bias}")
        
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.mr_window = mr_window
        self.mr_entry_z = mr_entry_z
        self.mr_exit_z = mr_exit_z
        self.mom_lookback = mom_lookback
        self.mom_threshold = mom_threshold
        self.use_seasonality = use_seasonality
        
        # Seasonality
        self.winter_months = set(winter_months)
        self.summer_months = set(summer_months)
        self.shoulder_months = set(shoulder_months)
        self.winter_bias = winter_bias
        self.summer_bias = summer_bias
        self.shoulder_bias = shoulder_bias
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate energy seasonal signals.
        
        Args:
            df: DataFrame with at least 'Close' column and DatetimeIndex
        
        Returns:
            DataFrame with added columns:
                - Returns: Log returns
                - Volatility: Rolling volatility (annualized)
                - HighVol: Boolean for high volatility regime
                - MR_Z: Z-score for mean reversion
                - MR_Signal: Mean reversion signal (-1/0/+1)
                - Mom_Return: Lookback momentum return
                - Mom_Signal: Momentum signal (-1/0/+1)
                - Month: Month number (1-12)
                - Season: Season name (Winter/Summer/Shoulder)
                - Seasonal_Bias: Seasonal bias value (-1 to +1)
                - Hybrid_Signal: Combined signal before seasonality
                - Signal: Final signal after seasonal filtering
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
        
        df = df.copy()
        
        # Calculate returns and volatility
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(window=self.vol_window).std() * np.sqrt(252)
        df['HighVol'] = (df['Volatility'] > self.vol_threshold).astype(int)
        
        # === MEAN REVERSION COMPONENT (High Volatility) ===
        # Calculate z-score
        rolling_mean = df['Close'].rolling(window=self.mr_window).mean()
        rolling_std = df['Close'].rolling(window=self.mr_window).std()
        df['MR_Z'] = (df['Close'] - rolling_mean) / rolling_std
        
        # Mean reversion signals (vectorized with state tracking)
        df['MR_Signal'] = 0
        mr_signal = np.zeros(len(df))
        current_position = 0
        
        for i in range(self.mr_window, len(df)):
            z = df['MR_Z'].iloc[i]
            
            # Entry logic
            if current_position == 0:
                if z < -self.mr_entry_z:  # Oversold → Long
                    current_position = 1
                elif z > self.mr_entry_z:  # Overbought → Short
                    current_position = -1
            
            # Exit logic
            elif current_position == 1:  # Long position
                if z > -self.mr_exit_z or z > self.mr_entry_z:  # Reverted or stop loss
                    current_position = 0
            
            elif current_position == -1:  # Short position
                if z < self.mr_exit_z or z < -self.mr_entry_z:  # Reverted or stop loss
                    current_position = 0
            
            mr_signal[i] = current_position
        
        df['MR_Signal'] = mr_signal
        
        # === MOMENTUM COMPONENT (Low Volatility) ===
        df['Mom_Return'] = df['Close'].pct_change(self.mom_lookback)
        df['Mom_Signal'] = 0
        
        # Momentum signals (vectorized)
        df.loc[df['Mom_Return'] > self.mom_threshold, 'Mom_Signal'] = 1   # Uptrend
        df.loc[df['Mom_Return'] < -self.mom_threshold, 'Mom_Signal'] = -1  # Downtrend
        
        # === SEASONALITY COMPONENT ===
        df['Month'] = df.index.month
        df['Season'] = 'Shoulder'
        df['Seasonal_Bias'] = self.shoulder_bias
        
        # Assign seasons and biases
        df.loc[df['Month'].isin(self.winter_months), 'Season'] = 'Winter'
        df.loc[df['Month'].isin(self.winter_months), 'Seasonal_Bias'] = self.winter_bias
        
        df.loc[df['Month'].isin(self.summer_months), 'Season'] = 'Summer'
        df.loc[df['Month'].isin(self.summer_months), 'Seasonal_Bias'] = self.summer_bias
        
        # === HYBRID SIGNAL (Regime-based) ===
        # High vol → Use mean reversion, Low vol → Use momentum
        df['Hybrid_Signal'] = np.where(
            df['HighVol'] == 1,
            df['MR_Signal'],
            df['Mom_Signal']
        )
        
        # === APPLY SEASONALITY FILTER ===
        if self.use_seasonality:
            # Apply seasonal bias to filter signals
            df['Signal'] = 0
            
            for i in range(len(df)):
                signal = df['Hybrid_Signal'].iloc[i]
                bias = df['Seasonal_Bias'].iloc[i]
                
                # Only take signals aligned with seasonal bias
                if signal == 1:  # Long signal
                    # Allow long if bias is positive or neutral
                    if bias >= 0:
                        df.loc[df.index[i], 'Signal'] = 1
                    elif bias < 0 and bias > -0.5:  # Weak bearish bias, still allow
                        df.loc[df.index[i], 'Signal'] = 1
                
                elif signal == -1:  # Short signal
                    # Allow short if bias is negative or neutral
                    if bias <= 0:
                        df.loc[df.index[i], 'Signal'] = -1
                    elif bias > 0 and bias < 0.5:  # Weak bullish bias, still allow
                        df.loc[df.index[i], 'Signal'] = -1
        else:
            # No seasonality filtering
            df['Signal'] = df['Hybrid_Signal']
        
        # Set warmup period to flat
        warmup = max(self.vol_window, self.mr_window, self.mom_lookback)
        df.iloc[:warmup, df.columns.get_loc('Signal')] = 0
        
        return df


class EnergySeasonalLongOnly(SignalModel):
    """
    Long-only version of EnergySeasonalSignal.
    
    Converts short signals to flat (for portfolios that don't support shorting).
    Maintains all the seasonal and regime-detection logic.
    """
    
    def __init__(self, **kwargs):
        """Initialize with same parameters as EnergySeasonalSignal."""
        self.base_signal = EnergySeasonalSignal(**kwargs)
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate long-only energy seasonal signals.
        
        Converts all short signals (-1) to flat (0).
        """
        df = self.base_signal.generate(df)
        
        # Convert shorts to flat
        df['Signal'] = df['Signal'].clip(lower=0)
        
        return df
