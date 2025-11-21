"""Paper trading simulator with risk controls and trade tracking."""
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from analysis.metrics import sharpe_ratio, max_drawdown

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
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        transaction_cost: float = 3,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        max_hold_days: Optional[int] = None,
        stop_mode: str = "close",
        max_position_pct: float = 1.0,
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

        # support per-row size overrides (signal may supply a `Size` column with fraction);
        # otherwise use global `max_position_pct` as the cap (1.0 = fully invested)
        if "Size" in df.columns:
            exec_size = df["Size"].shift(1).fillna(1.0).clip(upper=max_position_pct)
        else:
            exec_size = pd.Series(max_position_pct, index=df.index)

        # apply sizing to executed position
        df["ExecPosition"] = df["ExecPosition"] * exec_size

        # 1b. Apply execution-level risk controls (stop-loss, take-profit, max-hold)
        # Work on a copy of the executed position series so we can modify exits
        exec_series = df["ExecPosition"].copy()

        # mapping from (entry_date, exit_date) -> exit_reason for later trade annotation
        # use index labels (timestamps) so the mapping survives date filtering/slicing
        exit_reasons_map = {}

        # intraday exit price map: index_label -> exit_price (used when stop_mode != 'close')
        intraday_exit_map = {}

        current_pos = 0
        entry_idx = None
        entry_price = None
        hold_days = 0
        per_stop = None
        per_take = None
        per_max_hold = None

        # iterate through rows to detect stop / take-profit / max-hold triggers
        for i in range(len(df)):
            idx = df.index[i]
            price = df.at[idx, "Close"]
            pos = float(exec_series.iloc[i]) if not pd.isna(exec_series.iloc[i]) else 0.0

            # detect new entry
            if current_pos == 0 and pos != 0:
                current_pos = pos
                entry_idx = i
                entry_price = price if price is not None else None
                hold_days = 0
                # per-trade overrides from signal dataframe, if present
                per_stop = None
                per_take = None
                per_max_hold = None
                if "StopLossPct" in df.columns and pd.notna(df.at[idx, "StopLossPct"]):
                    try:
                        per_stop = float(df.at[idx, "StopLossPct"])
                    except Exception:
                        per_stop = None
                if "TakeProfitPct" in df.columns and pd.notna(df.at[idx, "TakeProfitPct"]):
                    try:
                        per_take = float(df.at[idx, "TakeProfitPct"])
                    except Exception:
                        per_take = None
                if "MaxHoldDays" in df.columns and pd.notna(df.at[idx, "MaxHoldDays"]):
                    try:
                        per_max_hold = int(df.at[idx, "MaxHoldDays"])
                    except Exception:
                        per_max_hold = None
                # fall back to global defaults if per-trade not provided
                if per_stop is None:
                    per_stop = stop_loss_pct
                if per_take is None:
                    per_take = take_profit_pct
                if per_max_hold is None:
                    per_max_hold = max_hold_days

            # if currently in a trade, evaluate stop/take/max-hold
            elif current_pos != 0:
                hold_days += 1
                triggered = False
                reason = None
                if entry_price is not None:
                    # determine which price to use for trigger detection depending on mode
                    check_price = price
                    if stop_mode == "low" and "Low" in df.columns:
                        check_price = df.at[idx, "Low"]
                    if stop_mode == "open" and "Open" in df.columns:
                        check_price = df.at[idx, "Open"]

                    if check_price is not None:
                        pnl = (check_price / entry_price - 1) * np.sign(current_pos)
                        if per_stop is not None and pnl <= -float(per_stop):
                            triggered = True
                            reason = "stop_loss"
                        if not triggered and per_take is not None and pnl >= float(per_take):
                            triggered = True
                            reason = "take_profit"
                if not triggered and per_max_hold is not None and per_max_hold > 0 and hold_days >= int(per_max_hold):
                    triggered = True
                    reason = "max_hold"

                if triggered:
                    # apply exit by setting tomorrow's executed position to 0 (if exists)
                    entry_label = df.index[entry_idx] if entry_idx is not None else None
                    trigger_label = df.index[i]  # Day when stop/take/max-hold triggered
                    
                    # if using intraday mode, record exit price so we can adjust today's strategy
                    if stop_mode in ("low", "open"):
                        # assume fill at the stop price level (entry*(1-stop)) for long, symmetric for short
                        if current_pos > 0:
                            exit_price = entry_price * (1 - float(per_stop)) if per_stop is not None else df.at[idx, "Close"]
                        else:
                            exit_price = entry_price * (1 + float(per_stop)) if per_stop is not None else df.at[idx, "Close"]
                        intraday_exit_map[trigger_label] = exit_price
                    
                    # Record exit reason with BOTH trigger date and actual exit date for robust matching
                    if i + 1 < len(df):
                        exec_series.iloc[i + 1] = 0
                        actual_exit_label = df.index[i + 1]  # Actual exit date (next day)
                        # Store with both keys for robust lookup
                        exit_reasons_map[(entry_label, trigger_label)] = reason
                        exit_reasons_map[(entry_label, actual_exit_label)] = reason
                    else:
                        # last row: mark exit on this index
                        exit_reasons_map[(entry_label, trigger_label)] = reason
                    
                    # reset current trade state
                    current_pos = 0
                    entry_idx = None
                    entry_price = None
                    hold_days = 0
                    per_stop = None
                    per_take = None
                    per_max_hold = None

        # assign adjusted executed positions back to dataframe
        df["ExecPosition"] = exec_series

        # 2. Daily returns
        df["Return"] = df["Close"].pct_change().fillna(0)

        # 3. Strategy returns before costs
        df["Strategy"] = df["ExecPosition"] * df["Return"]

        # If we recorded intraday exits (stop_mode in low/open), override the day's strategy
        # return on the trigger day to reflect execution at the stop price instead of close-to-close
        if intraday_exit_map:
            for exit_label, exit_price in intraday_exit_map.items():
                try:
                    # find previous close
                    loc = df.index.get_loc(exit_label)
                    if loc == 0:
                        continue
                    prev_close = df["Close"].iloc[loc - 1]
                    entry_pos = df["ExecPosition"].iloc[loc]
                    # intraday pnl from prev_close to exit_price
                    intraday_ret = (exit_price / prev_close - 1) * entry_pos
                    df.at[exit_label, "Strategy"] = intraday_ret
                    # Also set the day's Return to reflect exit price (used nowhere else but helpful to inspect)
                    df.at[exit_label, "Return"] = exit_price / prev_close - 1
                except Exception:
                    continue

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
        for trade_id, group in df[entry_mask | (df.index == df.index[-1])].groupby(
            "TradeID"
        ):
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

            # Annotate exit reason with improved matching logic
            exit_reason = None
            try:
                # Direct lookup (most common case)
                exit_reason = exit_reasons_map.get((entry_date, exit_date), None)
                
                # Fallback 1: Check the day before exit_date (for i+1 offset)
                if exit_reason is None and len(df.index) > 1:
                    prev_date_idx = df.index.get_loc(exit_date)
                    if prev_date_idx > 0:
                        prev_date = df.index[prev_date_idx - 1]
                        exit_reason = exit_reasons_map.get((entry_date, prev_date), None)
                
                # Fallback 2: Match by entry_date alone (last resort)
                if exit_reason is None:
                    for (k_entry, k_exit), r in exit_reasons_map.items():
                        if k_entry == entry_date:
                            exit_reason = r
                            break
            except Exception as e:
                # If all fails, leave as None (natural signal exit)
                exit_reason = None

            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "side": "long" if entry_pos > 0 else "short",
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_reason,
                }
            )

        self.trades = pd.DataFrame(trades)
        if not self.trades.empty:
            self.trades = self.trades[
                self.trades["entry_date"]
                >= (
                    pd.to_datetime(start_date)
                    if start_date
                    else self.trades["entry_date"].min()
                )
            ]

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
            trade_stats.update(
                {
                    "n_trades": int(n),
                    "win_rate": len(wins) / n,
                    "avg_win_pct": wins["pnl_pct"].mean() if not wins.empty else 0.0,
                    "avg_loss_pct": (
                        losses["pnl_pct"].mean() if not losses.empty else 0.0
                    ),
                    "gross_profit_pct": (
                        wins["pnl_pct"].sum() if not wins.empty else 0.0
                    ),
                    "gross_loss_pct": (
                        losses["pnl_pct"].sum() if not losses.empty else 0.0
                    ),
                }
            )
        else:
            stats = {
                k: 0
                for k in [
                    "n_trades",
                    "win_rate",
                    "avg_win_pct",
                    "avg_loss_pct",
                    "gross_profit_pct",
                    "gross_loss_pct",
                ]
            }
            stats["n_trades"] = 0

        # Final clean dict with rounded floats
        summary_ = {
            "total_return_pct": round(float(total_return), round_digits),
            "sharpe": round(float(sharpe), round_digits),
            "max_drawdown_pct": round(float(max_dd), round_digits),
            **{
                k: round(float(v), round_digits) if k != "n_trades" else int(v)
                for k, v in trade_stats.items()
            },
        }

        return summary_

    def print_summary(self):
        s = self.summary()
        print("\n" + "=" * 50)
        print("           BACKTEST SUMMARY")
        print("=" * 50)
        print(f"Total Return       : {s['total_return_pct']:+.2%}")
        print(f"Sharpe Ratio       : {s['sharpe']:+.3f}")
        print(f"Max Drawdown       : {s['max_drawdown_pct']:.2%}")
        print(f"Number of Trades   : {s['n_trades']}")
        if s["n_trades"] > 0:
            print(f"Win Rate           : {s['win_rate']:.1%}")
            print(f"Avg Win            : {s['avg_win_pct']:+.2%}")
            print(f"Avg Loss           : {s['avg_loss_pct']:+.2%}")
        print("=" * 50 + "\n")

    def plot(
        self,
        title: str = "Strategy Equity Curve",
        show: bool = True,
        save: bool = False,
    ):
        if not hasattr(self, "result"):
            raise RuntimeError("Run simulate() first")
        df = self.result

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        ax1.plot(df.index, df["PortfolioValue"], label="Strategy", linewidth=1.5)
        ax1.set_title(title)
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        df["ExecPosition"].plot(
            ax=ax2, drawstyle="steps-post", linewidth=1.5, color="tab:orange"
        )
        ax2.set_title("Position Over Time")
        ax2.set_ylabel("Position")
        ax2.set_ylim(-1.1, 1.1)

        plt.tight_layout()
        if save:
            plt.savefig(
                LOG_DIR
                / f"equity_curve_ensemble_{datetime.now().strftime('%Y%m%d')}.png"
            )

        if show:
            plt.show()

        return fig
