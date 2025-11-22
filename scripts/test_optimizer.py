"""
Test Parameter Optimization Framework
Demonstrates grid search and random search on MomentumSignalV2
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.optimizer import ParameterOptimizer, save_optimization_results
from core.backtest_engine import run_walk_forward
from signals.momentum import MomentumSignalV2
from core.paper_trader import PaperTrader


# Global trial counter for unique folder names
trial_counter = 0

def objective_function(params: dict) -> dict:
    """
    Run backtest with given parameters and return metrics.
    
    This is what the optimizer will call for each parameter combination.
    Each trial gets its own folder with full results and HTML report.
    """
    global trial_counter
    trial_counter += 1
    
    # Load data
    data = pd.read_csv('Dataset/spx_full_1990_2025.csv', index_col=0, parse_dates=True)
    
    # Create signal factory with params
    def signal_factory():
        return MomentumSignalV2(
            lookback=params['lookback'],
            entry_threshold=params['entry_threshold'],
            exit_threshold=params['exit_threshold']
        )
    
    # Create unique folder for this trial
    trial_name = f"trial_{trial_counter:03d}_lb{params['lookback']}_sl{int(params['stop_loss_pct']*100)}"
    trial_dir = f'logs/optimization/trials/{trial_name}'
    
    # Run walk-forward backtest with risk params
    try:
        results = run_walk_forward(
            signal_factory=signal_factory,
            df=data,
            train_size=int(len(data) * 0.6),  # 60% train (~14 years)
            test_size=int(len(data) * 0.2),   # 20% test (~5 years)
            lookback=params['lookback'],
            initial_cash=100000,
            stop_loss_pct=params['stop_loss_pct'],
            take_profit_pct=params.get('take_profit_pct'),
            max_hold_days=params.get('max_hold_days'),
            save_dir=trial_dir,  # Each trial gets its own folder with HTML report
            transaction_cost=3.0
        )
        
        # Extract metrics from overall summary
        overall = results.get('overall', {})
        
        return {
            'sharpe': overall.get('Sharpe', np.nan),
            'total_return': overall.get('Total Return (%)', np.nan),
            'max_drawdown': overall.get('Max Drawdown (%)', np.nan),
            'win_rate': overall.get('Win Rate (%)', np.nan),
            'num_trades': overall.get('Num Trades', 0)
        }
        
    except Exception as e:
        import traceback
        print(f"Error in trial {trial_counter}: {e}")
        print(traceback.format_exc())
        return {
            'sharpe': np.nan,
            'total_return': np.nan,
            'max_drawdown': np.nan,
            'win_rate': np.nan,
            'num_trades': 0
        }


def test_grid_search_small():
    """Test grid search with small parameter space"""
    global trial_counter
    trial_counter = 0  # Reset counter
    
    print("=" * 80)
    print("TEST 1: Small Grid Search (4 combinations)")
    print("Each trial will have its own folder in logs/optimization/trials/")
    print("=" * 80)
    
    param_grid = {
        'lookback': [200, 250],
        'entry_threshold': [0.02],
        'exit_threshold': [-0.01],
        'stop_loss_pct': [0.08, 0.10]
    }
    
    optimizer = ParameterOptimizer(
        objective_fn=objective_function,
        param_grid=param_grid,
        metric='sharpe',
        maximize=True,
        verbose=True
    )
    
    best_params, best_score, results_df = optimizer.grid_search()
    
    # Save results
    save_optimization_results(
        best_params=best_params,
        best_score=best_score,
        results_df=results_df,
        output_dir='logs/optimization/small_grid'
    )
    
    print("\nResults summary:")
    print(results_df.sort_values('sharpe', ascending=False))
    

def test_grid_search_full():
    """Test grid search with larger parameter space"""
    print("\n" + "=" * 80)
    print("TEST 2: Full Grid Search (testing multiple thresholds)")
    print("=" * 80)
    
    param_grid = {
        'entry_threshold': [0.01, 0.02, 0.03, 0.4],
        'exit_threshold': [-0.01, -0.02, -0.03],
        'lookback': [100, 150, 200, 250, 300],
        'stop_loss_pct': [0.05, 0.08, 0.10, 0.12, 0.15]
    }
    
    optimizer = ParameterOptimizer(
        objective_fn=objective_function,
        param_grid=param_grid,
        metric='sharpe',
        maximize=True,
        verbose=True
    )
    
    best_params, best_score, results_df = optimizer.grid_search()
    
    # Save results
    save_optimization_results(
        best_params=best_params,
        best_score=best_score,
        results_df=results_df,
        output_dir='logs/optimization/full_grid'
    )
    
    print("\nTop 5 results:")
    print(results_df.sort_values('sharpe', ascending=False).head())


def test_random_search():
    """Test random search"""
    print("\n" + "=" * 80)
    print("TEST 3: Random Search (20 iterations)")
    print("=" * 80)
    
    param_grid = {
        'lookback': [150, 200, 250, 300],
        'entry_threshold': [0.01, 0.015, 0.02, 0.025, 0.03],
        'exit_threshold': [-0.005, -0.01, -0.015, -0.02],
        'stop_loss_pct': [0.05, 0.08, 0.10, 0.12, 0.15]
    }
    
    optimizer = ParameterOptimizer(
        objective_fn=objective_function,
        param_grid=param_grid,
        metric='sharpe',
        maximize=True,
        verbose=True
    )
    
    best_params, best_score, results_df = optimizer.random_search(
        n_iter=20,
        random_state=42
    )
    
    # Save results
    save_optimization_results(
        best_params=best_params,
        best_score=best_score,
        results_df=results_df,
        output_dir='logs/optimization/random_search'
    )
    
    print("\nTop 5 results:")
    print(results_df.sort_values('sharpe', ascending=False).head())


if __name__ == '__main__':
    # Start with small grid search
    #test_grid_search_small()
    
    # Uncomment to run more extensive tests:
    test_grid_search_full()
    # test_random_search()
    
    print("\n" + "=" * 80)
    print("Optimization tests complete!")
    print("Check logs/optimization/ for detailed results")
    print("=" * 80)
