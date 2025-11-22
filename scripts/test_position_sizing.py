"""Test and compare different position sizing methods.

This script compares:
1. Fixed sizing (baseline)
2. Volatility targeting
3. Kelly criterion
4. Signal strength-based sizing

Each method is tested with walk-forward validation to see
how position sizing impacts Sharpe ratio and drawdown.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from signals.momentum import MomentumSignalV2
from core.backtest_engine import run_walk_forward
from core.position_sizers import (
    FixedSizing,
    VolatilityTargeting,
    KellyCriterion,
    SignalStrengthSizing,
)


def add_position_sizing(df, signal_df, sizer, sizer_name):
    """Apply position sizer to signal dataframe.
    
    Args:
        df: Original price dataframe
        signal_df: Signal dataframe with Position column
        sizer: Position sizer object
        sizer_name: Name for logging
        
    Returns:
        DataFrame with Size column added
    """
    print(f"\nApplying {sizer_name}...")
    
    # Create a merged dataframe with both price and signal data
    merged = df.copy()
    merged["Position"] = signal_df["Position"]
    
    # For signal strength sizing, add the strength column if available
    if isinstance(sizer, SignalStrengthSizing):
        if "Momentum" in signal_df.columns:
            merged["SignalStrength"] = signal_df["Momentum"].abs()
        else:
            print(f"  ⚠️  Warning: No signal strength column found, using fixed sizing")
            merged["SignalStrength"] = pd.Series(1.0, index=merged.index)
    
    # Calculate sized positions
    try:
        sized_positions = sizer.calculate_size(merged)
        
        # Add Size column (PaperTrader will use this)
        result = signal_df.copy()
        result["Size"] = sized_positions.abs()  # Size is always positive
        
        # Log statistics
        avg_size = result["Size"].mean()
        max_size = result["Size"].max()
        min_size = result[result["Size"] > 0]["Size"].min() if (result["Size"] > 0).any() else 0
        
        print(f"  Average size: {avg_size:.2f}")
        print(f"  Size range: {min_size:.2f} to {max_size:.2f}")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error applying {sizer_name}: {e}")
        return signal_df


def test_position_sizing_comparison():
    """Compare different position sizing methods."""
    
    print("=" * 70)
    print("POSITION SIZING COMPARISON TEST")
    print("=" * 70)
    
    # Load data
    data_path = PROJECT_ROOT / "Dataset" / "spx_1990_2025.csv"
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    
    # Define signal factory (simple momentum)
    def signal_factory(**kwargs):
        return MomentumSignalV2(
            lookback=kwargs.get('lookback', 100),
            entry_threshold=kwargs.get('entry_threshold', 0.02),
            exit_threshold=kwargs.get('exit_threshold', -0.01),
            sma_filter=kwargs.get('sma_filter', 100),
        )
    
    # Best params from previous optimization
    best_params = {
        'lookback': 60,
        'entry_threshold': 0.01,
        'exit_threshold': -0.005,
        'sma_filter': 50,
        'stop_loss_pct': 0.10,
    }
    
    # Walk-forward setup
    train_size = 2500  # ~10 years of training data
    test_size = 1250   # ~5 years of test data
    lookback = 200     # Warm-up period
    
    print(f"\nWalk-forward setup:")
    print(f"  Train size: {train_size} days (~10 years)")
    print(f"  Test size: {test_size} days (~5 years)")
    print(f"  Lookback: {lookback} days")
    print(f"  Signal parameters: {best_params}")
    
    # Define position sizers to test
    sizers = {
        "Fixed (baseline)": FixedSizing(size=1.0),
        "Vol Target 12%": VolatilityTargeting(
            target_vol=0.12,
            lookback=20,
            max_leverage=2.0,
        ),
        "Vol Target 8%": VolatilityTargeting(
            target_vol=0.08,
            lookback=20,
            max_leverage=1.5,
        ),
        "Vol Target 16%": VolatilityTargeting(
            target_vol=0.16,
            lookback=20,
            max_leverage=3.0,
        ),
    }
    
    results = {}
    
    # Test each position sizer
    for sizer_name, sizer in sizers.items():
        print("\n" + "=" * 70)
        print(f"Testing: {sizer_name}")
        print("=" * 70)
        
        # Create custom signal factory that includes position sizing
        def sized_signal_factory(**kwargs):
            signal = signal_factory(**kwargs)
            
            # Wrap the generate method to add sizing
            original_generate = signal.generate
            
            def generate_with_sizing(df_input):
                signal_df = original_generate(df_input)
                return add_position_sizing(df_input, signal_df, sizer, sizer_name)
            
            signal.generate = generate_with_sizing
            return signal
        
        # Run backtest with this sizer
        try:
            result = run_walk_forward(
                signal_factory=sized_signal_factory,
                df=df,
                train_size=train_size,
                test_size=test_size,
                lookback=lookback,
                initial_cash=100_000,
                transaction_cost=3.0,
                save_dir=f"logs/sizing_test/{sizer_name.replace(' ', '_').lower()}",
                stop_loss_pct=best_params.get('stop_loss_pct'),
                optimize_per_fold=False,
            )
            
            results[sizer_name] = result
            
            # Print summary
            print(f"\n{sizer_name} Results:")
            print(f"  Total Return: {result['overall']['total_return_pct']*100:.2f}%")
            print(f"  Sharpe Ratio: {result['overall']['sharpe']:.3f}")
            print(f"  Max Drawdown: {result['overall']['max_drawdown']:.2%}")
            print(f"  Number of Folds: {result['overall']['n_folds']}")
            
        except Exception as e:
            print(f"\n❌ Error testing {sizer_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison table
    print("\n\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Method':<20} {'Return %':>12} {'Sharpe':>10} {'Max DD':>10}")
    print("-" * 70)
    
    for sizer_name, result in results.items():
        print(
            f"{sizer_name:<20} "
            f"{result['overall']['total_return_pct']*100:>11.2f}% "
            f"{result['overall']['sharpe']:>10.3f} "
            f"{result['overall']['max_drawdown']:>10.2%}"
        )
    
    # Find best method
    if results:
        best_sharpe = max(results.items(), key=lambda x: x[1]['overall']['sharpe'])
        best_return = max(results.items(), key=lambda x: x[1]['overall']['total_return_pct'])
        
        print(f"\n🏆 Best Sharpe: {best_sharpe[0]} ({best_sharpe[1]['overall']['sharpe']:.3f})")
        print(f"💰 Best Return: {best_return[0]} ({best_return[1]['overall']['total_return_pct']*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("Test complete! Results saved to logs/sizing_test/")
    print("=" * 70)


if __name__ == "__main__":
    test_position_sizing_comparison()
