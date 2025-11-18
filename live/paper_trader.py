import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from utils.metrics import sharpe_ratio, max_drawdown

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LOG_DIR = PROJECT_ROOT / "logs"

class PaperTrader:
    """Simple paper trading simulator.

    Assumptions:
    - Signals are in a `Position` column with values {1, 0, -1} representing long, flat, short.
    - Executions use the previous day's signal (i.e., `Position.shift(1)`) to avoid lookahead.
    - No transaction costs/slippage by default (can be added later).
    - Fully invests the portfolio for each position (no partial sizing), but PnL scales with position (-1 short).
    """

    def __init__(self, initial_cash: float = 100_000):
        self.initial_cash = initial_cash

    def simulate(
        self,
        df: pd.DataFrame,
        position_col: str = "Position",
        start_date: "pd.Timestamp|str" = None,
        end_date: "pd.Timestamp|str" = None,
        transaction_cost:float=3
    ) -> pd.DataFrame:
        """Simulate strategy over `df`.

        Args:
            df: DataFrame with price and signal columns.
            position_col: column name that holds signal positions (1/0/-1).
            start_date: if provided, only compute portfolio and report trades from this date onward.
            end_date: if provided, restrict simulation to this end date (inclusive).
            transaction_cost: cost per trade in basis points (bps), e.g., 3 = 0.03%
        """
        df = df.copy()
        # Ensure position column exists
        if position_col not in df.columns:
            raise ValueError(f"Position column '{position_col}' not found in dataframe")

        # 1. Use executed position = yesterday's signal
        df["ExecPosition"] = df[position_col].shift(1).fillna(0)

        # 2. Daily returns 
        df["Return"] = df["Close"].pct_change().fillna(0)
        
        # 3. Strategy returns before costs
        df["Strategy"] = df["ExecPosition"] * df["Return"]
        
        # 4. Add transaction costs
        if transaction_cost > 0:
            # How much of the portfolio changed position today?
            df["PositionChange"] = df["ExecPosition"].diff().abs()
            df["PositionChange"] = df["PositionChange"].fillna(df["ExecPosition"].abs())

            # Apply cost proportional to turnover
            trade_cost = df["PositionChange"] * (transaction_cost / 10_000)
            df["Strategy"] -= trade_cost.fillna(0)
            
        

        # 5. Date filtering
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            df.loc[df.index < start_date, "Strategy"] = 0
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            df = df.loc[df.index <= end_date]

        # 6. Portfolio value computation
        df["PortfolioValue"] = (1 + df["Strategy"]).cumprod() * self.initial_cash
        """print(f"Strategy returns range: min {df['Strategy'].min():.4f}, max {df['Strategy'].max():.4f}")
        print(f"Any returns > 0.5? { (df['Strategy'] > 0.5).sum() } days")
        exit()"""

        # 7. Vectorized trade detection (this is the magic)
        df["Trade"] = df["ExecPosition"].diff().fillna(df["ExecPosition"])
        entry_mask = df["Trade"] != 0
        df["TradeID"] = entry_mask.cumsum()

        trades = []
        for trade_id, group in df[entry_mask | (df.index == df.index[-1])].groupby("TradeID"):
            if trade_id == 0:
                continue
            rows = df[df["TradeID"] == trade_id]
            if len(rows) == 0:
                continue

            entry_date = rows.index[0]
            entry_price = rows["Close"].iloc[0]
            entry_pos = rows["ExecPosition"].iloc[0]

            exit_date = rows.index[-1]
            exit_price = rows["Close"].iloc[-1]

            if entry_price == 0:
                continue

            pnl_pct = (exit_price / entry_price - 1) * np.sign(entry_pos)

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "side": "long" if entry_pos > 0 else "short",
                "pnl_pct": pnl_pct,
            })

        self.trades = pd.DataFrame(trades)
        if not self.trades.empty:
            self.trades = self.trades[self.trades["entry_date"] >= (pd.to_datetime(start_date) if start_date else self.trades["entry_date"].min())]

        self.result = df
        return df

    def summary(self, round_digits: int = 3) -> dict:
        if not hasattr(self, "result"):
            raise RuntimeError("No simulation run. Call simulate() first.")

        df = self.result

        # Core metrics
        total_return = df["PortfolioValue"].iloc[-1] / self.initial_cash - 1
        sharpe = sharpe_ratio(df["Strategy"].fillna(0))
        max_dd = max_drawdown(df["Strategy"].fillna(0))

        # Trade stats
        trade_stats = {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "gross_profit_pct": 0.0,
            "gross_loss_pct": 0.0,
        }

        trades = getattr(self, "trades", pd.DataFrame())
        if not trades.empty:
            wins = trades[trades["pnl_pct"] > 0]
            losses = trades[trades["pnl_pct"] <= 0]

            n = len(trades)
            trade_stats.update({
                "n_trades": int(n),
                "win_rate": len(wins) / n,
                "avg_win_pct": wins["pnl_pct"].mean() if not wins.empty else 0.0,
                "avg_loss_pct": losses["pnl_pct"].mean() if not losses.empty else 0.0,
                "gross_profit_pct": wins["pnl_pct"].sum() if not wins.empty else 0.0,
                "gross_loss_pct": losses["pnl_pct"].sum() if not losses.empty else 0.0,
            })
        else:
            stats = {k: 0 for k in ["n_trades", "win_rate", "avg_win_pct", "avg_loss_pct", "gross_profit_pct", "gross_loss_pct"]}
            stats["n_trades"] = 0
            
        # Final clean dict with rounded floats
        summary_ = {
            "total_return_pct": round(float(total_return), round_digits),
            "sharpe": round(float(sharpe), round_digits),
            "max_drawdown_pct": round(float(max_dd), round_digits),
            **{k: round(float(v), round_digits) if k != "n_trades" else int(v)
            for k, v in trade_stats.items()},
        }

        return summary_
    
    def print_summary(self):
        s = self.summary()
        print("\n" + "="*50)
        print("           BACKTEST SUMMARY")
        print("="*50)
        print(f"Total Return       : {s['total_return_pct']:+.2%}")
        print(f"Sharpe Ratio       : {s['sharpe']:+.3f}")
        print(f"Max Drawdown       : {s['max_drawdown_pct']:.2%}")
        print(f"Number of Trades   : {s['n_trades']}")
        if s['n_trades'] > 0:
            print(f"Win Rate           : {s['win_rate']:.1%}")
            print(f"Avg Win            : {s['avg_win_pct']:+.2%}")
            print(f"Avg Loss           : {s['avg_loss_pct']:+.2%}")
        print("="*50 + "\n")

    def plot(self, title: str = "Strategy Equity Curve", show: bool = True, save:bool=False):
        if not hasattr(self, "result"):
            raise RuntimeError("Run simulate() first")
        df = self.result

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(df.index, df["PortfolioValue"], label="Strategy", linewidth=1.5)
        ax1.set_title(title)
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        df["ExecPosition"].plot(ax=ax2, drawstyle="steps-post", linewidth=1.5, color="tab:orange")
        ax2.set_title("Position Over Time")
        ax2.set_ylabel("Position")
        ax2.set_ylim(-1.1, 1.1)

        plt.tight_layout()
        if save:
            plt.savefig(LOG_DIR / f"equity_curve_ensemble_{datetime.now().strftime('%Y%m%d')}.png")
            
        if show:
            plt.show()
            
        
        return fig

