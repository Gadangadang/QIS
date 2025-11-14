import pandas as pd  
from signals.base import SignalModel

class MomentumSignal(SignalModel):
    def __init__(self, lookback=20):
        self.lookback = lookback

    def generate(self, df:pd.DataFrame)->pd.DataFrame:
        df = df.copy()
        df["Momentum"] = df["Close"].pct_change(self.lookback)
        df["Position"] = df["Momentum"].apply(lambda x: 1 if x > 0 else -1)
        return df 

