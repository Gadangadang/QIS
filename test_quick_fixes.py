"""Quick fixes based on diagnostic insights.

This script tests improved configurations that address the critical issues:
1. Disable toxic short trades
2. Add regime filter (bull market only)
3. Adjust parameters based on diagnostics
"""
import pandas as pd
from core.backtest_engine import run_walk_forward
from signals.momentum import MomentumSignal
from signals.ensemble import EnsembleSignalNew
from analysis.diagnostics import ModelDiagnostics
from pathlib import Path


def test_original_broken_model():
    """Baseline: Original momentum signal (for comparison)."""
    print("\n" + "=" * 70)
    print("BASELINE: Original Momentum Signal (Broken)")
    print("=" * 70)
    
    df = pd.read_csv("Dataset/spx_full_1990_2025.csv", index_col=0, parse_dates=True)
    df = df.sort_index()
    
    results = run_walk_forward(
        signal_factory=lambda: MomentumSignal(lookback=120, threshold=0.02),
        df=df,
        train_size=int(len(df) * 0.6),
        test_size=int(len(df) * 0.2),
        lookback=250,
        initial_cash=100_000,
        transaction_cost=3.0,
        stop_loss_pct=0.10,
        save_dir='logs/fix_baseline',
    )
    
    return results


def test_long_only_momentum():
    """FIX 1: Force long-only (disable toxic shorts)."""
    print("\n" + "=" * 70)
    print("FIX 1: Long-Only Momentum")
    print("=" * 70)
    
    df = pd.read_csv("Dataset/spx_full_1990_2025.csv", index_col=0, parse_dates=True)
    df = df.sort_index()
    
    # Use ensemble signal which has bull market filter built in
    results = run_walk_forward(
        signal_factory=lambda: EnsembleSignalNew(),  # Has 200-day MA filter
        df=df,
        train_size=int(len(df) * 0.6),
        test_size=int(len(df) * 0.2),
        lookback=250,
        initial_cash=100_000,
        transaction_cost=3.0,
        stop_loss_pct=0.10,
        save_dir='logs/fix_long_only',
    )
    
    return results


def test_reduced_stops():
    """FIX 2: Wider stops (original 10% might be too tight given win/loss asymmetry)."""
    print("\n" + "=" * 70)
    print("FIX 2: Long-Only + Wider Stops (15%)")
    print("=" * 70)
    
    df = pd.read_csv("Dataset/spx_full_1990_2025.csv", index_col=0, parse_dates=True)
    df = df.sort_index()
    
    results = run_walk_forward(
        signal_factory=lambda: EnsembleSignalNew(),
        df=df,
        train_size=int(len(df) * 0.6),
        test_size=int(len(df) * 0.2),
        lookback=250,
        initial_cash=100_000,
        transaction_cost=3.0,
        stop_loss_pct=0.15,  # Wider stops
        save_dir='logs/fix_wider_stops',
    )
    
    return results


def test_no_stops():
    """FIX 3: No stops at all (let signal decide exits)."""
    print("\n" + "=" * 70)
    print("FIX 3: Long-Only + No Stops (Signal-Driven Exits)")
    print("=" * 70)
    
    df = pd.read_csv("Dataset/spx_full_1990_2025.csv", index_col=0, parse_dates=True)
    df = df.sort_index()
    
    results = run_walk_forward(
        signal_factory=lambda: EnsembleSignalNew(),
        df=df,
        train_size=int(len(df) * 0.6),
        test_size=int(len(df) * 0.2),
        lookback=250,
        initial_cash=100_000,
        transaction_cost=3.0,
        stop_loss_pct=None,  # No stops
        save_dir='logs/fix_no_stops',
    )
    
    return results


def compare_fixes():
    """Run all fixes and compare results."""
    print("\n" + "🔬" * 35)
    print("RUNNING DIAGNOSTIC-DRIVEN FIXES")
    print("🔬" * 35)
    
    results_dict = {}
    
    # Run all tests
    try:
        results_dict['baseline'] = test_original_broken_model()
    except Exception as e:
        print(f"Baseline failed: {e}")
        results_dict['baseline'] = None
    
    try:
        results_dict['long_only'] = test_long_only_momentum()
    except Exception as e:
        print(f"Long-only failed: {e}")
        results_dict['long_only'] = None
    
    try:
        results_dict['wider_stops'] = test_reduced_stops()
    except Exception as e:
        print(f"Wider stops failed: {e}")
        results_dict['wider_stops'] = None
    
    try:
        results_dict['no_stops'] = test_no_stops()
    except Exception as e:
        print(f"No stops failed: {e}")
        results_dict['no_stops'] = None
    
    # Compare results
    print("\n" + "=" * 90)
    print("COMPARISON: Impact of Diagnostic-Driven Fixes")
    print("=" * 90)
    print(f"{'Configuration':<25} {'Total Return':>15} {'Sharpe':>10} {'Max DD':>10} {'Trades':>8} {'Win Rate':>10}")
    print("-" * 90)
    
    for name, results in results_dict.items():
        if results is None:
            print(f"{name:<25} {'FAILED':>15}")
            continue
        
        overall = results['overall']
        trades = results['trades']
        
        total_ret = overall.get('total_return_pct', 0)
        sharpe = overall.get('sharpe', 0)
        max_dd = overall.get('max_drawdown', 0)
        n_trades = len(trades) if not trades.empty else 0
        win_rate = (trades['pnl_pct'] > 0).mean() if not trades.empty else 0
        
        print(f"{name:<25} {total_ret:>14.2%} {sharpe:>10.3f} {max_dd:>10.2%} {n_trades:>8} {win_rate:>10.1%}")
    
    print("=" * 90)
    
    # Detailed diagnostics for best performer
    print("\n" + "🏆" * 35)
    print("DETAILED DIAGNOSTICS: Best Configuration")
    print("🏆" * 35)
    
    # Find best by Sharpe
    best_name = None
    best_sharpe = -999
    for name, results in results_dict.items():
        if results and results['overall'].get('sharpe', -999) > best_sharpe:
            best_sharpe = results['overall'].get('sharpe', -999)
            best_name = name
    
    if best_name:
        print(f"\nBest: {best_name.upper()} (Sharpe: {best_sharpe:+.3f})")
        diag = ModelDiagnostics(results_dict[best_name])
        print(diag.summary_report())
        
        # Save best diagnostics
        Path(f'logs/fix_{best_name}').mkdir(parents=True, exist_ok=True)
        diag.save_report(f'logs/fix_{best_name}/diagnostics_detailed.txt')
    else:
        print("\n⚠ All configurations failed!")
    
    print("\n" + "=" * 90)
    print("KEY INSIGHTS FROM FIXES:")
    print("=" * 90)
    print("1. If long-only helps: Shorts are toxic → disable permanently or fix signal")
    print("2. If wider/no stops help: Stop-loss killing edge → adjust or remove")
    print("3. If all fail similarly: Signal has no edge → need better signal/features")
    print("4. Compare Sharpe improvement → quantify value of each fix")
    print("=" * 90)
    
    return results_dict


if __name__ == "__main__":
    results = compare_fixes()
