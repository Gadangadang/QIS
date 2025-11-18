import pandas as pd
import numpy as np 
from signals.base import SignalModel
from signals.mean_reversion import MeanReversionSignal
from signals.momentum import MomentumSignal

class EnsembleSignal(SignalModel):
    def __init__(self):
        self.signals = [
            #MomentumSignal(lookback=60,  threshold=0.025),
            #MomentumSignal(lookback=120, threshold=0.02),
            #MomentumSignal(lookback=250, threshold=0.018),
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
        df["EnsemblePosition"] = positions_df.mean(axis=1).round().clip(-1, 1).astype(int)
        # Only go long when above 200-day MA. Never short. Ever.
        df["TrendFilter"] = (df["Close"] > df["Close"].rolling(200).mean()).astype(int)
        df["Position"] = df["EnsemblePosition"] * df["TrendFilter"]  # zero out when below
        
        return df
    
    
    
class EnsembleSignalNew(SignalModel):
    def __init__(self):
        pass
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Trend filter — the only thing that matters long-term
        df["SMA200"] = df["Close"].rolling(200).mean()
        df["InBullMarket"] = df["Close"] > df["SMA200"]
        
        # 2. Only use mean-reversion signals (they are fast and high-Sharpe)
        mr10 = MeanReversionSignal(window=10, entry_z=2.2, exit_z=1.0).generate(df.copy())
        mr20 = MeanReversionSignal(window=20, entry_z=2.0, exit_z=1.0).generate(df.copy())
        mr60 = MeanReversionSignal(window=60, entry_z=1.8, exit_z=1.0).generate(df.copy())
        
        # 3. Combine MR signals (majority vote)
        df["MR_Sum"] = (mr10["Position"] + mr20["Position"] + mr60["Position"])
        df["MR_Vote"] = np.sign(df["MR_Sum"])  # -1, 0, or +1
        
        # 4. FINAL POSITION: only trade MR when in bull market
        df["Position"] = df["MR_Vote"].where(df["InBullMarket"], 0)
        
        # 5. Burn-in
        df.iloc[:200, df.columns.get_loc("Position")] = 0
        
        return df