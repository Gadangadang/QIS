import pandas as pd
import numpy as np
from signals.base import SignalModel


class MeanReversionSignal(SignalModel):
    def __init__(self, window=20, entry_z=2.0, exit_z=0.5):
        self.window = window
        self.entry_z = entry_z  # Only trade at ~2σ deviation
        self.exit_z = exit_z  # Optional: exit early when back to mean

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"]
        sma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        df["Z"] = (close - sma) / std

        # Initialize position column
        df["Position"] = 0
        
        # Build positions bar-by-bar
        for i in range(self.window, len(df)):
            prev_position = df.iloc[i - 1]["Position"]
            z_score = df.iloc[i]["Z"]
            
            # If flat, check for entry
            if prev_position == 0:
                if z_score <= -self.entry_z:
                    df.iloc[i, df.columns.get_loc("Position")] = 1  # Long entry
                elif z_score >= self.entry_z:
                    df.iloc[i, df.columns.get_loc("Position")] = -1  # Short entry
                else:
                    df.iloc[i, df.columns.get_loc("Position")] = 0  # Stay flat
            
            # If holding long, check for exit
            elif prev_position == 1:
                if z_score >= -self.exit_z:  # Mean reversion: exit when Z crosses back toward zero
                    df.iloc[i, df.columns.get_loc("Position")] = 0
                elif z_score > self.entry_z:  # Stop loss: price moved against us
                    df.iloc[i, df.columns.get_loc("Position")] = 0
                else:
                    df.iloc[i, df.columns.get_loc("Position")] = 1  # Hold
            
            # If holding short, check for exit
            elif prev_position == -1:
                if z_score <= self.exit_z:  # Mean reversion: exit when Z crosses back toward zero
                    df.iloc[i, df.columns.get_loc("Position")] = 0
                elif z_score < -self.entry_z:  # Stop loss: price moved against us
                    df.iloc[i, df.columns.get_loc("Position")] = 0
                else:
                    df.iloc[i, df.columns.get_loc("Position")] = -1  # Hold

        df["Position"] = df["Position"].astype(int)
        return df
