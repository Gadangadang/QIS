"""Run diagnostics on walk-forward outputs and save results.

Usage (from repo root):

    python3 -m backtest.run_diagnostics

Options:
    --combined      Path to `combined_returns.csv` (default `logs/walkforward/combined_returns.csv`)
    --data          Market data CSV (default `Dataset/spx_full_1990_2025.csv`)
    --trades-glob   Glob pattern for trades files (default `logs/walkforward/trades_fold_*.csv`)
    --out           Output diagnostics file (default `logs/walkforward/diagnostics.txt`)
    --top-n         How many worst days/trades to show (default 30)

The script uses only Python stdlib (`csv`, `datetime`, `glob`) so it runs in minimal environments.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import glob
from typing import List, Tuple


def read_combined(path: Path) -> List[Tuple[datetime.date, float]]:
    rows = []
    with path.open('r') as f:
        r = csv.reader(f)
        hdr = next(r, None)
        for row in r:
            if not row:
                continue
            try:
                d = datetime.strptime(row[0], '%Y-%m-%d').date()
            except Exception:
                try:
                    d = datetime.fromisoformat(row[0]).date()
                except Exception:
                    continue
            try:
                val = float(row[1])
            except Exception:
                try:
                    val = float(row[1].strip())
                except Exception:
                    continue
            rows.append((d, val))
    return rows


def read_market_close(path: Path) -> Tuple[List[Tuple[datetime.date, float]], dict]:
    data = []
    with path.open('r') as f:
        r = csv.reader(f)
        hdr = next(r, [])
        idx = {h: i for i, h in enumerate(hdr)}
        close_idx = idx.get('Close', 1)
        for row in r:
            if not row:
                continue
            try:
                d = datetime.strptime(row[0], '%Y-%m-%d').date()
            except Exception:
                try:
                    d = datetime.fromisoformat(row[0]).date()
                except Exception:
                    continue
            try:
                c = float(row[close_idx])
            except Exception:
                c = None
            data.append((d, c))
    data.sort()
    close_map = {d: c for d, c in data}
    # compute pct-change map
    prev = None
    ret = {}
    for d, c in data:
        if prev is None:
            ret[d] = None
        else:
            if c is None or prev[1] is None:
                ret[d] = None
            else:
                try:
                    ret[d] = (c - prev[1]) / prev[1]
                except Exception:
                    ret[d] = None
        prev = (d, c)
    return data, ret


def read_trades(glob_pattern: str) -> List[dict]:
    out = []
    for p in sorted(glob.glob(glob_pattern)):
        path = Path(p)
        try:
            with path.open('r') as f:
                r = csv.reader(f)
                hdr = next(r, [])
                idx = {h: i for i, h in enumerate(hdr)}
                for row in r:
                    if not row:
                        continue
                    rec = {
                        'file': str(path),
                        'entry_date': row[idx['entry_date']] if 'entry_date' in idx else '',
                        'exit_date': row[idx['exit_date']] if 'exit_date' in idx else '',
                        'entry_price': row[idx['entry_price']] if 'entry_price' in idx else '',
                        'exit_price': row[idx['exit_price']] if 'exit_price' in idx else '',
                        'side': row[idx['side']] if 'side' in idx else '',
                        'pnl_pct': None,
                    }
                    if 'pnl_pct' in idx:
                        try:
                            rec['pnl_pct'] = float(row[idx['pnl_pct']])
                        except Exception:
                            rec['pnl_pct'] = None
                    out.append(rec)
        except Exception:
            continue
    return out


def write_diagnostics(out_path: Path, worst_days: List[Tuple], stitched_summary: str, worst_trades: List[dict], trades_on_worst: List[dict]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as fo:
        def W(s=''):
            fo.write(str(s) + '\n')
            print(s)
        W('=== Walk-forward diagnostics ===')
        W('\n-- Worst strategy days (date, strategy_return, market_return) --')
        for d, v, mr in worst_days:
            W(f"{d}  {v:.6f}  market={mr}")
        W('\n-- Stitched equity summary --')
        W(stitched_summary)
        W('\n-- Worst trades (file,entry->exit,side,pnl_pct) --')
        for t in worst_trades:
            W(f"{t['file']}  {t['entry_date']} -> {t['exit_date']}  {t['side']}  {t['pnl_pct']}")
        W('\n-- Trades with entry/exit on worst strategy dates (exact match) --')
        if trades_on_worst:
            for t in trades_on_worst:
                W(f"{t['file']}  {t['entry_date']} -> {t['exit_date']}  {t['side']}  pnl_pct={t['pnl_pct']}")
        else:
            W('No trades had exact entry/exit on the worst strategy dates')
        W('\nDiagnostics saved to: ' + str(out_path))


def main():
    p = argparse.ArgumentParser(description='Run walk-forward diagnostics')
    p.add_argument('--combined', default='logs/walkforward/combined_returns.csv')
    p.add_argument('--data', default='Dataset/spx_full_1990_2025.csv')
    p.add_argument('--trades-glob', default='logs/walkforward/trades_fold_*.csv')
    p.add_argument('--stitched', default='logs/walkforward/stitched_equity.csv')
    p.add_argument('--out', default='logs/walkforward/diagnostics.txt')
    p.add_argument('--top-n', type=int, default=30)
    args = p.parse_args()

    combined_path = Path(args.combined)
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined returns not found: {combined_path}")

    combined = read_combined(combined_path)
    combined_sorted = sorted(combined, key=lambda x: x[1])
    top_n = min(args.top_n, len(combined_sorted))
    worst = combined_sorted[:top_n]

    # market
    data_path = Path(args.data)
    if data_path.exists():
        _, market_ret = read_market_close(data_path)
    else:
        market_ret = {}

    worst_days = [(d, v, market_ret.get(d)) for d, v in worst]

    # stitched summary
    stitched_path = Path(args.stitched)
    stitched_summary = 'stitched_equity.csv not found'
    if stitched_path.exists():
        try:
            with stitched_path.open('r') as f:
                r = csv.reader(f)
                next(r, None)
                rows = list(r)
                if rows:
                    stitched_summary = f"stitched_start: {rows[0][1]}  stitched_end: {rows[-1][1]}  n={len(rows)}"
        except Exception:
            stitched_summary = 'failed to read stitched_equity.csv'

    # trades
    trades = read_trades(args.trades_glob)
    trades_with_pnl = [t for t in trades if t['pnl_pct'] is not None]
    trades_with_pnl_sorted = sorted(trades_with_pnl, key=lambda x: x['pnl_pct'])
    worst_trades = trades_with_pnl_sorted[:args.top_n]

    worst_dates_set = set([d for d, _ in worst])
    trades_on_worst = []
    for t in trades:
        for key in ('entry_date', 'exit_date'):
            if t.get(key):
                try:
                    dt = datetime.strptime(t[key], '%Y-%m-%d').date()
                except Exception:
                    continue
                if dt in worst_dates_set:
                    trades_on_worst.append(t)
                    break

    write_diagnostics(Path(args.out), worst_days, stitched_summary, worst_trades, trades_on_worst)


if __name__ == '__main__':
    main()
