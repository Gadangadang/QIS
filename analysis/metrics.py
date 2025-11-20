"""Performance metrics for backtesting analysis."""
import numpy as np
import pandas as pd


def sharpe_ratio(returns, risk_free=0.04):
    """Calculate annualized Sharpe ratio from daily returns.

    Args:
        returns (pd.Series or np.ndarray): daily strategy returns (not cumulative).
        risk_free (float): annual risk-free rate (default 0.04 = 4%).

    Returns:
        float: annualized Sharpe ratio.
    """
    if len(returns) == 0:
        return 0.0
    excess = returns - risk_free / 252
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return np.sqrt(252) * excess.mean() / std


def sortino_ratio(returns, risk_free=0.04, target_return=0.0):
    """Calculate annualized Sortino ratio from daily returns.

    Uses downside deviation instead of total standard deviation.

    Args:
        returns (pd.Series or np.ndarray): daily strategy returns (not cumulative).
        risk_free (float): annual risk-free rate (default 0.04 = 4%).
        target_return (float): target return threshold (default 0.0).

    Returns:
        float: annualized Sortino ratio.
    """
    if len(returns) == 0:
        return 0.0
    excess = returns - risk_free / 252
    downside = returns[returns < target_return]
    if len(downside) == 0:
        return 0.0
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return np.sqrt(252) * excess.mean() / downside_std


def max_drawdown(returns):
    """Compute maximum drawdown from daily returns.

    Args:
        returns (pd.Series or np.ndarray): daily strategy returns (not cumulative).

    Returns:
        float: most negative drawdown (e.g., -0.10 means -10%).
    """
    if len(returns) == 0:
        return 0.0
    # compute cumulative wealth index
    cum = (1 + returns.fillna(0)).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return drawdown.min()


def cagr(returns):
    """Calculate Compound Annual Growth Rate from daily returns.

    Args:
        returns (pd.Series or np.ndarray): daily strategy returns (not cumulative).

    Returns:
        float: annualized return (CAGR).
    """
    if len(returns) == 0:
        return 0.0
    cum = (1 + returns.fillna(0)).cumprod()
    total_return = cum.iloc[-1] - 1 if isinstance(cum, pd.Series) else cum[-1] - 1
    n_days = len(returns)
    n_years = n_days / 252.0
    if n_years == 0:
        return 0.0
    return (1 + total_return) ** (1 / n_years) - 1


def calmar_ratio(returns):
    """Calculate Calmar ratio (CAGR / abs(MaxDrawdown)).

    Args:
        returns (pd.Series or np.ndarray): daily strategy returns (not cumulative).

    Returns:
        float: Calmar ratio.
    """
    if len(returns) == 0:
        return 0.0
    cagr_val = cagr(returns)
    max_dd = abs(max_drawdown(returns))
    if max_dd == 0:
        return 0.0
    return cagr_val / max_dd


def profit_factor(trades_df):
    """Calculate profit factor from trades DataFrame.

    Args:
        trades_df (pd.DataFrame): DataFrame with 'pnl_pct' column.

    Returns:
        float: profit factor (gross_profit / abs(gross_loss)).
    """
    if trades_df is None or trades_df.empty or 'pnl_pct' not in trades_df.columns:
        return 0.0

    wins = trades_df[trades_df['pnl_pct'] > 0]
    losses = trades_df[trades_df['pnl_pct'] <= 0]

    gross_profit = wins['pnl_pct'].sum() if not wins.empty else 0.0
    gross_loss = abs(losses['pnl_pct'].sum()) if not losses.empty else 0.0

    if gross_loss == 0:
        return 0.0 if gross_profit == 0 else np.inf

    return gross_profit / gross_loss


def win_rate(trades_df):
    """Calculate win rate from trades DataFrame.

    Args:
        trades_df (pd.DataFrame): DataFrame with 'pnl_pct' column.

    Returns:
        float: win rate (0.0 to 1.0).
    """
    if trades_df is None or trades_df.empty or 'pnl_pct' not in trades_df.columns:
        return 0.0

    wins = trades_df[trades_df['pnl_pct'] > 0]
    return len(wins) / len(trades_df)


def average_win(trades_df):
    """Calculate average winning trade percentage.

    Args:
        trades_df (pd.DataFrame): DataFrame with 'pnl_pct' column.

    Returns:
        float: average win percentage.
    """
    if trades_df is None or trades_df.empty or 'pnl_pct' not in trades_df.columns:
        return 0.0

    wins = trades_df[trades_df['pnl_pct'] > 0]
    return wins['pnl_pct'].mean() if not wins.empty else 0.0


def average_loss(trades_df):
    """Calculate average losing trade percentage.

    Args:
        trades_df (pd.DataFrame): DataFrame with 'pnl_pct' column.

    Returns:
        float: average loss percentage (negative value).
    """
    if trades_df is None or trades_df.empty or 'pnl_pct' not in trades_df.columns:
        return 0.0

    losses = trades_df[trades_df['pnl_pct'] <= 0]
    return losses['pnl_pct'].mean() if not losses.empty else 0.0
