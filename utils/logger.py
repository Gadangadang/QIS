import pandas as pd  
import os
from datetime import date      

def log_daily_result(df:pd.DataFrame, ticker:str, log_path:str="logs/daily_log.csv"):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        ticker (str): _description_
        log_path (str, optional): _description_. Defaults to "logs/daily_log.csv".
    """
   
    latest = df.iloc[-1]
    log_entry = {
        "Date": latest.name.date() if hasattr(latest.name, "date") else date.today(),
        "Ticker": ticker,
        "Signal": latest.get("Signal", None),
        "Position": latest.get("Position", None),
        "Return": latest.get("Return", None),
        "StrategyReturn": latest.get("Strategy", None),
        "CumulativeReturn": df["Strategy"].cumsum().iloc[-1]
    }
    
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
    else:
        log_df = pd.DataFrame(columns=log_entry.keys())
        
    log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(f"Logged result {ticker} on {log_entry.get('Date')}")