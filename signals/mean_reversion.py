import pandas as pd 
from signals.base import SignalModel

class MeanReversionSignal(SignalModel):
    def __init__(self, window=20):
        self.window = window 
        
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["SMA"] = df["Close"].rolling(self.window).mean()
        df["Signal"] = -1*(df["Close"]-df["SMA"])
        df["Position"] = df["Signal"].apply(lambda x: 1 if x > 0 else -1)
        return df 