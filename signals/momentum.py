import pandas as pd  

def generate_momentum_signal(df:pd.DataFrame, lookback: int=20)->pd.DataFrame:
    df = df.copy()
    df["Momentum"] = df["Close"].pct_change(lookback)
    df["Position"] = df["Momentum"].apply(lambda x: 1 if x > 0 else -1)
    return df 

