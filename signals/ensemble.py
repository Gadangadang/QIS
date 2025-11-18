import pandas as pd
import numpy as np 
from signals.base import SignalModel
from signals.mean_reversion import MeanReversionSignal
from signals.momentum import MomentumSignal

class EnsembleSignal(SignalModel):
    def __init__(self):
        self.signals = [
            MomentumSignal(lookback=60,  threshold=0.025),
            MomentumSignal(lookback=120, threshold=0.02),
            MomentumSignal(lookback=250, threshold=0.018),
            MeanReversionSignal(window=10,  entry_z=2.2, exit_z=1.0),
            MeanReversionSignal(window=20,  entry_z=2.0, exit_z=1.0),
            MeanReversionSignal(window=60,  entry_z=1.8, exit_z=1.0),
        ]
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        position_dfs = []
        
        for signal in self.signals:
            signaled = signal.generate(df.copy())  # ← clean, isolated
            position_dfs.append(signaled[["Position"]])
        
        # Combine
        positions_df = pd.concat(position_dfs, axis=1)
        df["Position"] = positions_df.mean(axis=1).round().clip(-1, 1).astype(int)
        
        return df