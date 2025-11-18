import pandas as pd
import matplotlib.pyplot as plt
from utils.metrics import sharpe_ratio, max_drawdown


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

        # Use executed position = yesterday's signal
        df["ExecPosition"] = df[position_col].shift(1).fillna(0)

        # Daily returns
        df["Return"] = df["Close"].pct_change().fillna(0)
        df["Strategy"] = df["ExecPosition"] * df["Return"]
        
        # Add transaction costs
        # === TRANSACTION COSTS ===
        if transaction_cost > 0:
            # How much of the portfolio changed position today?
            df["PositionChange"] = df["ExecPosition"].diff().abs()
            df["PositionChange"] = df["PositionChange"].fillna(df["ExecPosition"].abs())

            # Apply cost proportional to turnover
            trade_cost = df["PositionChange"] * (transaction_cost / 10_000)
            df["Strategy"] -= trade_cost.fillna(0)
            
        

        # Optionally restrict to an end date for the internal dataframe used for calculations
        if end_date is not None:
            end_ts = pd.to_datetime(end_date)
            df = df.loc[df.index <= end_ts].copy()

        # If a start_date is given, ensure Strategy before start_date does not affect portfolio
        s = df["Strategy"].fillna(0).copy()
        if start_date is not None:
            start_ts = pd.to_datetime(start_date)
            # Zero out strategy before the start date so cumulative product begins at start
            s.loc[df.index < start_ts] = 0

        # Portfolio value computed with possible zeroed pre-start strategy
        df["PortfolioValue"] = (1 + s).cumprod() * self.initial_cash

        # Track trades (entry / exit when ExecPosition changes)
        trades = []
        current_trade = None
        prev_pos = 0
        for idx, row in df.iterrows():
            pos = row["ExecPosition"]
            price = row.get("Close", None)

            if pos != prev_pos:
                # close previous trade if any
                if current_trade is not None:
                    current_trade["exit_date"] = idx
                    current_trade["exit_price"] = price
                    # compute pnl percent
                    if current_trade["entry_price"] and current_trade["exit_price"]:
                        if current_trade["entry_price"] != 0:
                            current_trade["pnl_pct"] = (
                                current_trade["exit_price"] / current_trade["entry_price"] - 1
                            ) * (1 if current_trade["entry_side"] == "long" else -1)
                        else:
                            current_trade["pnl_pct"] = None
                    trades.append(current_trade)
                    current_trade = None

                # open new trade if pos != 0
                if pos != 0:
                    current_trade = {
                        "entry_date": idx,
                        "entry_price": price,
                        "entry_side": "long" if pos == 1 else "short",
                        "exit_date": None,
                        "exit_price": None,
                        "pnl_pct": None,
                    }

            prev_pos = pos

        # if there's an open trade at the end, close it at last price
        if current_trade is not None:
            last_idx = df.index[-1]
            last_price = df.iloc[-1]["Close"]
            current_trade["exit_date"] = last_idx
            current_trade["exit_price"] = last_price
            if current_trade["entry_price"] and current_trade["exit_price"]:
                if current_trade["entry_price"] != 0:
                    current_trade["pnl_pct"] = (
                        current_trade["exit_price"] / current_trade["entry_price"] - 1
                    ) * (1 if current_trade["entry_side"] == "long" else -1)
            trades.append(current_trade)

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["pnl_pct"] = trades_df["pnl_pct"].astype(float)

        # If start_date provided, only keep trades that start on/after start_date
        if start_date is not None and not trades_df.empty:
            start_ts = pd.to_datetime(start_date)
            trades_df = trades_df[trades_df["entry_date"] >= start_ts].reset_index(drop=True)

        self.trades = trades_df
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

    def plot(self, show=True):
        if not hasattr(self, "result"):
            raise RuntimeError("No simulation run. Call simulate() first.")
        df = self.result
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax[0].plot(df.index, (1 + df["Strategy"]).cumprod() - 1)
        ax[0].set_title("Strategy Cumulative Return")
        ax[0].axhline(0, color="k", lw=0.5)

        ax[1].plot(df.index, df["ExecPosition"], drawstyle="steps-post")
        ax[1].set_title("Executed Position (1=long, -1=short, 0=flat)")

        plt.tight_layout()
        if show:
            plt.show()
        return fig
