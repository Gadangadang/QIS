import numpy as np


def sharpe_ratio(returns, risk_free=0.04):
    """Calculate annualized Sharpe ratio from daily returns.

    Args:
        returns (pd.Series or np.ndarray): daily strategy returns (not cumulative).
        risk_free (float): annual risk-free rate (default 0.04 = 4%).
    Returns:
        float: annualized Sharpe ratio.
    """
    excess = returns - risk_free / 252
    return np.sqrt(252) * excess.mean() / excess.std()


def max_drawdown(returns):
    """Compute maximum drawdown from daily returns.

    Args:
        returns (pd.Series or np.ndarray): daily strategy returns (not cumulative).

    Returns:
        float: most negative drawdown (e.g., -0.10 means -10%).
    """
    # compute cumulative wealth index
    cum = (1 + returns.fillna(0)).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return drawdown.min()
