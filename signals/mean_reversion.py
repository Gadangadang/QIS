import pandas as pd 

def generate_mean_reversion_signal(df: pd.DataFrame, window:int = 20) -> pd.DataFrame:
    df = df.copy()
    df["SMA"] = df["Close"].rolling(window).mean()
    df["Signal"] = -1*(df["Close"]-df["SMA"])
    df["Position"] = df["Signal"].apply(lambda x: 1 if x > 0 else -1)
    return df 

