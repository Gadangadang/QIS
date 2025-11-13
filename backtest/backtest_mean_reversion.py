# backtest/backtest_mean_reversion

import yfinance as yf 
import pandas as pd 
import matplotlib.pyplot as plt 
from signals.mean_reversion import generate_mean_reversion_signal

ticker="SPY"

df = yf.download(ticker, start="2015-01-01", end="2023-12-31")
df = generate_mean_reversion_signal(df)


df["Return"] = df["Adj Close"].pct_change()
df["Strategy"] = df["Position"].shift(1)*df["Return"]


#Evaluation
df[["Return", "Strategy"]].cumsum().plot(title="Cummulative returns")
plt.show()

sharpe = df["Strategy"].mean()/df["Strategy"].std()*(252**0.5)
print(f"Sharpe: {sharpe}")