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

        df["Position"] = 0

        # Entry
        long_entry = df["Z"] <= -self.entry_z
        short_entry = df["Z"] >= self.entry_z
        df.loc[long_entry, "Position"] = 1
        df.loc[short_entry, "Position"] = -1

        # Exit: when Z crosses back toward zero
        exit_long = (df["Position"] == 1) & (df["Z"] >= -self.exit_z)
        exit_short = (df["Position"] == -1) & (df["Z"] <= self.exit_z)
        df.loc[exit_long | exit_short, "Position"] = 0

        # Forward fill
        df["Position"] = df["Position"].replace(0, np.nan).ffill()

        # FINAL SAFETY: if price moves AGAINST position by > entry_z, exit immediately
        kill_long = (df["Position"] == 1) & (df["Z"] > self.entry_z)
        kill_short = (df["Position"] == -1) & (df["Z"] < -self.entry_z)
        df.loc[kill_long | kill_short, "Position"] = 0

        # Burn-in
        df.iloc[: self.window + 20, df.columns.get_loc("Position")] = 0

        df["Position"] = df["Position"].fillna(0).astype(int)
        return df
