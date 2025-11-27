"""Ensemble signal strategies combining multiple signal generators."""
import pandas as pd
import numpy as np
from signals.base import SignalModel
from signals.mean_reversion import MeanReversionSignal
from signals.momentum import MomentumSignal
from typing import List, Dict


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


class AdaptiveEnsemble(SignalModel):
    """
    Advanced ensemble that dynamically weights strategies based on recent performance.
    
    Key Features:
    1. Combines multiple strategy types (momentum, mean reversion, trend following)
    2. Calculates rolling Sharpe ratio for each strategy
    3. Dynamically adjusts weights based on what's working
    4. Includes volatility regime filter
    5. Signal strength threshold to avoid weak trades
    
    This ensemble aims to beat buy-and-hold by:
    - Adapting to changing market conditions
    - Weighting strategies that are currently performing
    - Staying flat when signals are weak or conflicting
    
    Example:
        >>> from signals.momentum import MomentumSignalV2
        >>> from signals.trend_following_long_short import TrendFollowingLongShort
        >>> 
        >>> strategies = [
        ...     ('momentum', MomentumSignalV2(lookback=60), 0.4),
        ...     ('trend_ls', TrendFollowingLongShort(), 0.6)
        ... ]
        >>> ensemble = AdaptiveEnsemble(strategies, method='adaptive')
        >>> signals = ensemble.generate(price_data)
    """
    
    def __init__(
        self,
        strategies: List[tuple],
        method: str = 'adaptive',
        adaptive_lookback: int = 60,
        signal_threshold: float = 0.3,
        rebalance_frequency: int = 20
    ):
        """
        Initialize adaptive ensemble signal generator.
        
        Args:
            strategies (List[tuple]): List of (name, SignalModel, initial_weight) tuples
            method (str): 'weighted_average', 'majority_vote', 'unanimous', or 'adaptive'
            adaptive_lookback (int): Rolling window for performance calculation
            signal_threshold (float): Min combined signal strength for position
            rebalance_frequency (int): How often to update adaptive weights
        """
        # Normalize weights
        total_weight = sum(w for _, _, w in strategies)
        self.strategies = [
            (name, signal_gen, w / total_weight) 
            for name, signal_gen, w in strategies
        ]
        
        self.method = method
        self.adaptive_lookback = adaptive_lookback
        self.signal_threshold = signal_threshold
        self.rebalance_frequency = rebalance_frequency
        self.strategy_names = [name for name, _, _ in self.strategies]
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, lookback: int) -> float:
        """Calculate rolling Sharpe ratio."""
        if len(returns) < lookback:
            return 0.0
        
        recent_returns = returns.iloc[-lookback:]
        mean_return = recent_returns.mean() * 252
        std_return = recent_returns.std() * np.sqrt(252)
        
        if std_return == 0 or np.isnan(std_return):
            return 0.0
        
        return mean_return / std_return
    
    def _update_adaptive_weights(
        self, 
        strategy_signals: Dict[str, pd.DataFrame],
        prices: pd.DataFrame,
        current_idx: int
    ) -> Dict[str, float]:
        """Calculate adaptive weights based on recent Sharpe ratios."""
        if current_idx < self.adaptive_lookback + 20:
            return {name: weight for name, _, weight in self.strategies}
        
        sharpe_ratios = {}
        
        for name, _, _ in self.strategies:
            signal_df = strategy_signals[name]
            signals = signal_df["Signal"].iloc[:current_idx]
            
            # Calculate strategy returns
            price_returns = prices["Close"].pct_change()
            strategy_returns = signals.shift(1) * price_returns
            
            sharpe = self._calculate_sharpe_ratio(strategy_returns, self.adaptive_lookback)
            sharpe_ratios[name] = max(sharpe, 0)  # No negative weights
        
        # Convert Sharpe ratios to weights
        total_sharpe = sum(sharpe_ratios.values())
        
        if total_sharpe == 0:
            weights = {name: 1.0 / len(self.strategies) for name in self.strategy_names}
        else:
            weights = {name: sharpe / total_sharpe for name, sharpe in sharpe_ratios.items()}
        
        return weights
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate adaptive ensemble signals.
        
        Returns DataFrame with individual strategy signals, weights, and final signal.
        """
        df = df.copy()
        
        # Generate signals from all strategies
        strategy_signals = {}
        for name, signal_gen, _ in self.strategies:
            strategy_df = signal_gen.generate(df.copy())
            strategy_signals[name] = strategy_df
            df[f"{name}_Signal"] = strategy_df["Signal"]
        
        # Combine signals based on method
        if self.method == 'weighted_average':
            df["CombinedSignal"] = 0.0
            for name, _, weight in self.strategies:
                df["CombinedSignal"] += strategy_signals[name]["Signal"] * weight
        
        elif self.method == 'majority_vote':
            df["CombinedSignal"] = sum(
                strategy_signals[name]["Signal"] for name, _, _ in self.strategies
            )
            df["CombinedSignal"] = np.sign(df["CombinedSignal"])
        
        elif self.method == 'unanimous':
            df["CombinedSignal"] = 0.0
            for i in range(len(df)):
                signals = [strategy_signals[name]["Signal"].iloc[i] for name, _, _ in self.strategies]
                if all(s == 1 for s in signals):
                    df.iloc[i, df.columns.get_loc("CombinedSignal")] = 1.0
                elif all(s == -1 for s in signals):
                    df.iloc[i, df.columns.get_loc("CombinedSignal")] = -1.0
                else:
                    df.iloc[i, df.columns.get_loc("CombinedSignal")] = 0.0
        
        elif self.method == 'adaptive':
            df["CombinedSignal"] = 0.0
            current_weights = {name: weight for name, _, weight in self.strategies}
            
            for i in range(len(df)):
                # Update weights periodically
                if i % self.rebalance_frequency == 0 and i > 0:
                    current_weights = self._update_adaptive_weights(strategy_signals, df, i)
                
                # Calculate weighted signal
                combined = sum(
                    strategy_signals[name]["Signal"].iloc[i] * current_weights[name]
                    for name, _, _ in self.strategies
                )
                df.iloc[i, df.columns.get_loc("CombinedSignal")] = combined
                
                # Store weights
                for name in self.strategy_names:
                    weight_col = f"{name}_Weight"
                    if weight_col not in df.columns:
                        df[weight_col] = 0.0
                    df.iloc[i, df.columns.get_loc(weight_col)] = current_weights[name]
        
        # Apply signal threshold
        df["SignalStrength"] = df["CombinedSignal"].abs()
        df["Signal"] = 0
        
        df.loc[df["CombinedSignal"] > self.signal_threshold, "Signal"] = 1
        df.loc[df["CombinedSignal"] < -self.signal_threshold, "Signal"] = -1
        df.loc[df["SignalStrength"] < self.signal_threshold, "Signal"] = 0
        
        # Forward fill
        df["Signal"] = df["Signal"].replace(0, np.nan).ffill().fillna(0)
        df["Signal"] = df["Signal"].astype(int)
        
        return df
