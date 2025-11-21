import pandas as pd
import numpy as np
from signals.base import SignalModel


class MomentumSignal(SignalModel):
    def __init__(self, lookback=20, threshold=0.02, exit_threshold=0.0):
        self.lookback = lookback
        self.threshold = threshold  # Only trade at ~2% momentum
        self.exit_threshold = exit_threshold  # Optional: exit early when momentum reverts
        

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
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
    def __init__(self, lookback=120, entry_threshold=0.02, exit_threshold=-0.01):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold  # Negative value for symmetric exits
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"]
        
        # Calculate momentum
        df["Momentum"] = close / close.shift(self.lookback) - 1
        
        # Regime filter (CRITICAL!)
        df["SMA200"] = close.rolling(200).mean()
        df["BullMarket"] = close > df["SMA200"]
        
        # Entry: Strong positive momentum in bull market
        enter = (df["Momentum"] > self.entry_threshold) & df["BullMarket"]
        
        # Exit: Either strong negative momentum OR bear market
        exit = (df["Momentum"] < self.exit_threshold) | ~df["BullMarket"]
        
        # Generate positions
        df["Position"] = 0
        df.loc[enter, "Position"] = 1
        df.loc[exit, "Position"] = 0
        
        # Forward fill (stay in position until exit trigger)
        df["Position"] = df["Position"].replace(0, np.nan).ffill().fillna(0)
        
        # Burn-in
        df.iloc[:self.lookback + 200, df.columns.get_loc("Position")] = 0
        
        return df