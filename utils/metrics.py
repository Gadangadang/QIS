import numpy as np

def sharpe_ratio(returns, risk_free=0.04):
    excess = returns - risk_free/252
    return np.sqrt(252)*excess.mean()/excess.std()

def max_drawndown(cum_returns):
    peak = cum_returns.cummax()
    drawdown =(cum_returns - peak)/peak 
    return drawdown.min()