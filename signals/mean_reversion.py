import pandas as pd 
import numpy as np 
from signals.base import SignalModel

class MeanReversionSignal(SignalModel):
    def __init__(self, window=20, entry_z=2.0, exit_z=0.5):
        self.window = window
        self.entry_z = entry_z    # Only trade at ~2σ deviation
        self.exit_z = exit_z      # Optional: exit early when back to mean

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        sma = df["Close"].rolling(self.window).mean()
        std = df["Close"].rolling(self.window).std()
        df["Z"] = (df["Close"] - sma) / std

        # Entry
        df["RawPosition"] = np.select(
            [df["Z"] <= -self.entry_z, df["Z"] >= self.entry_z],
            [1, -1], default=0
        )

        # Forward fill + exit when Z gets back toward mean
        df["Position"] = df["RawPosition"].replace(0, np.nan).ffill().fillna(0)
        exit_condition = (df["Position"] != 0) & (df["RawPosition"] == 0) & (abs(df["Z"]) < self.exit_z)
        df.loc[exit_condition, "Position"] = 0

        # New signal overrides exit
        df.loc[df["RawPosition"] != 0, "Position"] = df["RawPosition"]
        
        return df