"""Test diagnostics and exit_reason fixes."""
import pandas as pd
from core.backtest_engine import run_walk_forward
from signals.momentum import MomentumSignal, MomentumSignalV2
from analysis.diagnostics import ModelDiagnostics

def test_exit_reasons_and_diagnostics():
    """Test that exit_reason mapping works and diagnostics provide insights."""
    print("=" * 70)
    print("Testing Exit Reason Fixes & Model Diagnostics")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv("Dataset/spx_full_1990_2025.csv", index_col=0, parse_dates=True)
    df = df.sort_index()
    print(f"   ✓ Loaded {len(df):,} rows")
    
    # Run backtest with stop-loss to generate risk-exit trades
    print("\n2. Running backtest with stop-loss (10%)...")
    results = run_walk_forward(
        signal_factory=lambda: MomentumSignalV2(lookback=120, entry_threshold=0.02, exit_threshold=-0.01),
        df=df,
        train_size=int(len(df) * 0.6),
        test_size=int(len(df) * 0.2),
        lookback=250,
        initial_cash=100_000,
        transaction_cost=3.0,
        stop_loss_pct=0.10,  # 10% stop loss
        save_dir='logs/test_diagnostics',
    )
    print(f"   ✓ Completed {results['overall']['n_folds']} folds")
    
    # Check exit_reason coverage
    print("\n3. Checking exit_reason coverage...")
    trades = results['trades']
    if not trades.empty:
        exit_reasons = trades['exit_reason'].value_counts(dropna=False)
        print(f"   Total trades: {len(trades)}")
        print(f"   Exit reason distribution:")
        for reason, count in exit_reasons.items():
            reason_label = reason if pd.notna(reason) and reason != '' else 'signal_exit'
            pct = count / len(trades) * 100
            print(f"     {reason_label:15s}: {count:4d} trades ({pct:5.1f}%)")
        
        # Check if stop_loss exits exist
        stop_loss_trades = trades[trades['exit_reason'] == 'stop_loss']
        if len(stop_loss_trades) > 0:
            print(f"\n   ✓ SUCCESS: Found {len(stop_loss_trades)} stop-loss exits")
            print(f"     Average loss on stopped trades: {stop_loss_trades['pnl_pct'].mean():.2%}")
        else:
            print(f"\n   ⚠ WARNING: No stop-loss exits found (may be expected if no stops triggered)")
    
    # Run diagnostics
    print("\n4. Running model diagnostics...")
    diag = ModelDiagnostics(results)
    
    # Signal quality
    print("\n   A. Signal Quality:")
    signal_qual = diag.signal_quality_report()
    if 'error' not in signal_qual:
        print(f"      Directional accuracy: {signal_qual.get('signal_accuracy_pct', 0):.1%}")
        print(f"      Theoretical Sharpe:   {signal_qual.get('signal_sharpe_theoretical', 0):.3f}")
    
    # Execution leakage
    print("\n   B. Execution Leakage:")
    leakage = diag.execution_leakage()
    if 'error' not in leakage:
        print(f"      Theoretical Sharpe: {leakage.get('theoretical_sharpe', 0):+.3f}")
        print(f"      Actual Sharpe:      {leakage.get('actual_sharpe', 0):+.3f}")
        print(f"      Leakage:            {leakage.get('sharpe_leakage', 0):+.3f}")
        
        if leakage.get('sharpe_leakage', 0) > 0.5:
            print("      🚨 INSIGHT: Large execution leakage! Check:")
            print("         - Transaction costs too high?")
            print("         - Stop losses triggering too often?")
            print("         - Position sizing issues?")
    
    # Regime breakdown
    print("\n   C. Regime Performance:")
    regimes = diag.regime_breakdown()
    if not regimes.empty:
        for _, row in regimes.iterrows():
            print(f"      {row['regime']:15s}: Sharpe {row['sharpe']:+.3f}")
        
        # Check for regime issues
        bull = regimes[regimes['regime'] == 'Bull Market']
        bear = regimes[regimes['regime'] == 'Bear Market']
        if not bull.empty and not bear.empty:
            if bull['sharpe'].iloc[0] < 0 and bear['sharpe'].iloc[0] < 0:
                print("      🚨 INSIGHT: Negative Sharpe in BOTH regimes!")
                print("         Signal may not have predictive power")
    
    # Trade anatomy
    print("\n   D. Trade Anatomy:")
    anatomy = diag.trade_anatomy()
    if 'error' not in anatomy:
        print(f"      Win rate:           {anatomy.get('win_rate', 0):.1%}")
        print(f"      Avg hold (wins):    {anatomy.get('avg_hold_days_wins', 0):.1f} days")
        print(f"      Avg hold (losses):  {anatomy.get('avg_hold_days_losses', 0):.1f} days")
        print(f"      Long vs Short:")
        print(f"        Long win rate:    {anatomy.get('long_win_rate', 0):.1%}")
        print(f"        Short win rate:   {anatomy.get('short_win_rate', 0):.1%}")
        
        # Insights
        if anatomy.get('win_rate', 0) < 0.3:
            print("      🚨 INSIGHT: Very low win rate (<30%)")
            print("         Need larger wins or better entry timing")
        
        long_wr = anatomy.get('long_win_rate', 0)
        short_wr = anatomy.get('short_win_rate', 0)
        if abs(long_wr - short_wr) > 0.15:
            better = 'Long' if long_wr > short_wr else 'Short'
            print(f"      💡 INSIGHT: {better} trades perform much better")
            print(f"         Consider long-only or adjust signal for shorts")
    
    # Fold stability
    print("\n   E. Fold Stability:")
    stability = diag.fold_stability()
    if not stability.empty:
        sharpe_row = stability[stability['metric'] == 'sharpe']
        if not sharpe_row.empty:
            std = sharpe_row['std'].iloc[0]
            mean = sharpe_row['mean'].iloc[0]
            if std > abs(mean):
                print(f"      Sharpe std ({std:.3f}) > mean ({mean:.3f})")
                print("      🚨 INSIGHT: Very inconsistent performance across folds")
                print("         May indicate overfitting or regime-dependent signal")
    
    print("\n" + "=" * 70)
    print("✓ Test complete! Check logs/test_diagnostics/diagnostics.txt for full report")
    print("=" * 70)
    
    return results, diag


if __name__ == "__main__":
    results, diag = test_exit_reasons_and_diagnostics()
    
    # Print full diagnostics report
    print("\n\n" + diag.summary_report())
