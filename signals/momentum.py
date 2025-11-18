import pandas as pd  
import numpy as np 
from signals.base import SignalModel

class MomentumSignal(SignalModel):
    def __init__(self, lookback=20, threshold=0.02, exit_threshold=0.0):
        self.lookback = lookback
        self.threshold = threshold    # Only trade at ~2% momentum
        self.exit_threshold = exit_threshold      # Optional: exit early when momentum reverts

    def generate(self, df:pd.DataFrame)->pd.DataFrame:
        # Momentum = total return over lookback period
        df["Momentum"] = df["Close"].pct_change(self.lookback)
        
        # Raw entry signals
        entry_long = df["Momentum"] > self.threshold
        entry_short = df["Momentum"] < -self.threshold
        raw_position = np.where(entry_long, 1, np.where(entry_short, -1, 0))

        # Make a Series with the same index so assignments align correctly
        raw_s = pd.Series(raw_position, index=df.index)

        # Forward fill position, but exit when momentum weakens
        df["Position"] = raw_s.replace(0, np.nan).ffill()
        df["Position"] = df["Position"].fillna(0)
        
        # Exit condition: momentum fades
        fade_long = (df["Position"] == 1) & (df["Momentum"] < self.exit_threshold)
        fade_short = (df["Position"] == -1) & (df["Momentum"] > -self.exit_threshold)
        df.loc[fade_long | fade_short, "Position"] = 0
        
        # Override with new entries
        # Ensure assigning only the corresponding values (align by index)
        df.loc[raw_s != 0, "Position"] = raw_s[raw_s != 0]
        return df
      

