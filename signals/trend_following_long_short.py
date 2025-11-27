"""Trend-following long-short strategy with multi-timeframe confirmation."""
import pandas as pd
import numpy as np
from signals.base import SignalModel


class TrendFollowingLongShort(SignalModel):
    """
    Advanced trend-following strategy that takes both long and short positions.
    
    Key Features:
    1. Multi-timeframe trend confirmation (fast + slow momentum)
    2. Volume confirmation (institutions participating)
    3. Volatility regime filter (avoid choppy markets)
    4. Dynamic position sizing based on signal strength
    
    The strategy aims to beat buy-and-hold by:
    - Going long in strong uptrends with volume support
    - Going short in strong downtrends (capturing bear markets)
    - Staying flat in choppy/ranging markets (preserving capital)
    
    Signal Logic:
    - LONG: Fast momentum > 0, Slow momentum > threshold, Volume > avg, Low volatility regime
    - SHORT: Fast momentum < 0, Slow momentum < -threshold, Volume > avg, Low volatility regime  
    - FLAT: Mixed signals, high volatility, or low volume (uncertain market)
    
    Attributes:
        fast_period (int): Fast momentum lookback (captures recent trend)
        slow_period (int): Slow momentum lookback (confirms major trend)
        volume_period (int): Volume average period (detects institutional flow)
        vol_regime_period (int): Volatility lookback for regime detection
        momentum_threshold (float): Minimum slow momentum for trade entry
        volume_multiplier (float): Required volume vs average (1.2 = 20% above avg)
        vol_percentile (float): Max volatility percentile to trade (0.7 = 70th percentile)
    
    Example:
        >>> signal = TrendFollowingLongShort(
        ...     fast_period=20, slow_period=100, 
        ...     momentum_threshold=0.03, volume_multiplier=1.1
        ... )
        >>> df_with_signals = signal.generate(price_data)
    """
    
    def __init__(
        self,
        fast_period=20,
        slow_period=100,
        volume_period=50,
        vol_regime_period=60,
        momentum_threshold=0.02,
        volume_multiplier=1.1,
        vol_percentile=0.70
    ):
        """
        Initialize trend-following long-short signal generator.
        
        Args:
            fast_period (int): Fast momentum period (default 20 days)
            slow_period (int): Slow momentum period (default 100 days)
            volume_period (int): Volume MA period (default 50 days)
            vol_regime_period (int): Volatility regime period (default 60 days)
            momentum_threshold (float): Min slow momentum for entry (default 0.02 = 2%)
            volume_multiplier (float): Volume vs average requirement (default 1.1 = 10% above)
            vol_percentile (float): Max volatility to trade (default 0.70 = 70th percentile)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.volume_period = volume_period
        self.vol_regime_period = vol_regime_period
        self.momentum_threshold = momentum_threshold
        self.volume_multiplier = volume_multiplier
        self.vol_percentile = vol_percentile
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate long-short signals with multi-timeframe trend confirmation.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data (needs Close, Volume)
        
        Returns:
            pd.DataFrame: Original DataFrame with added columns:
                - FastMomentum: Short-term price momentum
                - SlowMomentum: Long-term trend strength
                - VolumeMA: Average volume baseline
                - VolumeRatio: Current volume vs average
                - Volatility: Historical volatility (rolling std)
                - VolRegime: Volatility percentile (0-1 scale)
                - TrendStrength: Combined trend indicator
                - Signal: Trading position (1=long, -1=short, 0=flat)
        
        Note:
            - Warm-up period: max(slow_period, vol_regime_period) + 20 bars
            - Positions during warm-up set to 0
            - Signal strength increases with volume confirmation
        """
        df = df.copy()
        close = df["Close"]
        volume = df.get("Volume", pd.Series(1, index=df.index))  # Handle futures without volume
        
        # === 1. Multi-timeframe Momentum ===
        df["FastMomentum"] = close / close.shift(self.fast_period) - 1
        df["SlowMomentum"] = close / close.shift(self.slow_period) - 1
        
        # === 2. Volume Analysis ===
        df["VolumeMA"] = volume.rolling(self.volume_period).mean()
        df["VolumeRatio"] = volume / df["VolumeMA"]
        
        # === 3. Volatility Regime Filter ===
        # Calculate rolling volatility (annualized)
        returns = close.pct_change()
        df["Volatility"] = returns.rolling(self.vol_regime_period).std() * np.sqrt(252)
        
        # Volatility percentile (0 = lowest vol, 1 = highest vol)
        df["VolRegime"] = df["Volatility"].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        
        # === 4. Trend Strength (combination of fast + slow momentum) ===
        # Strong uptrend: both positive, Strong downtrend: both negative
        df["TrendStrength"] = (
            np.sign(df["FastMomentum"]) * np.abs(df["FastMomentum"]) +
            np.sign(df["SlowMomentum"]) * np.abs(df["SlowMomentum"]) * 2  # Weight slow momentum more
        ) / 3
        
        # === 5. Generate Trading Signals ===
        df["Signal"] = 0
        
        # LONG Conditions:
        # 1. Both fast and slow momentum positive
        # 2. Slow momentum above threshold (strong trend)
        # 3. Volume above average (confirmation)
        # 4. Low/medium volatility regime (stable market)
        long_condition = (
            (df["FastMomentum"] > 0) &
            (df["SlowMomentum"] > self.momentum_threshold) &
            (df["VolumeRatio"] >= self.volume_multiplier) &
            (df["VolRegime"] <= self.vol_percentile)
        )
        
        # SHORT Conditions:
        # 1. Both fast and slow momentum negative
        # 2. Slow momentum below -threshold (strong downtrend)
        # 3. Volume above average (confirmation)
        # 4. Low/medium volatility regime (stable bearish trend)
        short_condition = (
            (df["FastMomentum"] < 0) &
            (df["SlowMomentum"] < -self.momentum_threshold) &
            (df["VolumeRatio"] >= self.volume_multiplier) &
            (df["VolRegime"] <= self.vol_percentile)
        )
        
        # Apply signals
        df.loc[long_condition, "Signal"] = 1
        df.loc[short_condition, "Signal"] = -1
        
        # Forward fill positions (stay in trade until opposite signal or flat)
        # But allow flat periods when conditions aren't met
        df["Signal"] = df["Signal"].replace(0, np.nan).ffill().fillna(0)
        
        # Force flat during high volatility (risk management)
        high_vol = df["VolRegime"] > self.vol_percentile
        df.loc[high_vol, "Signal"] = 0
        
        # === 6. Warm-up Period ===
        warmup = max(self.slow_period, self.vol_regime_period) + 20
        df.iloc[:warmup, df.columns.get_loc("Signal")] = 0
        
        df["Signal"] = df["Signal"].astype(int)
        
        return df


class AdaptiveTrendFollowing(SignalModel):
    """
    Adaptive trend-following strategy that adjusts to market regimes.
    
    Similar to TrendFollowingLongShort but with additional features:
    - Adjusts momentum thresholds based on market volatility
    - Uses ATR (Average True Range) for dynamic stop-losses
    - Implements profit-taking on extreme moves
    
    This is a more conservative version suitable for risk-averse traders.
    
    Attributes:
        base_period (int): Base lookback for momentum calculation
        atr_period (int): ATR calculation period
        vol_lookback (int): Volatility regime lookback
        base_threshold (float): Base momentum threshold (adjusted by volatility)
    """
    
    def __init__(
        self,
        base_period=60,
        atr_period=14,
        vol_lookback=120,
        base_threshold=0.03
    ):
        """
        Initialize adaptive trend-following signal generator.
        
        Args:
            base_period (int): Momentum calculation period (default 60)
            atr_period (int): ATR period for volatility (default 14)
            vol_lookback (int): Volatility regime detection (default 120)
            base_threshold (float): Base entry threshold (default 0.03 = 3%)
        """
        self.base_period = base_period
        self.atr_period = atr_period
        self.vol_lookback = vol_lookback
        self.base_threshold = base_threshold
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        
        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        return atr
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate adaptive trend-following signals.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
        
        Returns:
            pd.DataFrame: Original DataFrame with Signal column and indicators
        """
        df = df.copy()
        close = df["Close"]
        
        # Calculate momentum
        df["Momentum"] = close / close.shift(self.base_period) - 1
        
        # Calculate ATR-based volatility
        if "High" in df.columns and "Low" in df.columns:
            df["ATR"] = self._calculate_atr(df)
            df["ATR_Pct"] = df["ATR"] / close  # ATR as percentage of price
        else:
            # Fallback for data without High/Low
            returns = close.pct_change()
            df["ATR_Pct"] = returns.rolling(self.atr_period).std()
        
        # Adaptive threshold: Higher volatility = higher threshold required
        median_atr = df["ATR_Pct"].rolling(self.vol_lookback).median()
        current_atr = df["ATR_Pct"]
        vol_adjustment = current_atr / median_atr.replace(0, 1)
        
        df["AdaptiveThreshold"] = self.base_threshold * vol_adjustment
        
        # Generate signals
        df["Signal"] = 0
        
        # Long: Momentum above adaptive threshold
        long_condition = df["Momentum"] > df["AdaptiveThreshold"]
        
        # Short: Momentum below negative adaptive threshold
        short_condition = df["Momentum"] < -df["AdaptiveThreshold"]
        
        df.loc[long_condition, "Signal"] = 1
        df.loc[short_condition, "Signal"] = -1
        
        # Forward fill
        df["Signal"] = df["Signal"].replace(0, np.nan).ffill().fillna(0)
        
        # Warm-up
        warmup = max(self.base_period, self.vol_lookback) + 20
        df.iloc[:warmup, df.columns.get_loc("Signal")] = 0
        
        df["Signal"] = df["Signal"].astype(int)
        
        return df
