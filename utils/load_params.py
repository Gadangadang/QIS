"""
Helper functions to load optimized parameters.
Use in notebooks or scripts to load best parameters from optimization runs.
"""
import json
from pathlib import Path


def load_best_params(params_file: str = None) -> dict:
    """
    Load best parameters from optimization run.
    
    Args:
        params_file: Path to best_params.json (default: auto-detect from current directory)
        
    Returns:
        dict with parameter values
        
    Example:
        >>> params = load_best_params()
        >>> signal = MomentumSignalV2(
        ...     lookback=params['lookback'],
        ...     entry_threshold=params['entry_threshold'],
        ...     exit_threshold=params['exit_threshold']
        ... )
    """
    if params_file is None:
        # Try to find the file relative to current working directory
        cwd = Path.cwd()
        # Check if we're in notebooks/ directory
        if cwd.name == 'notebooks':
            params_file = '../logs/optimization/best_params.json'
        else:
            params_file = 'logs/optimization/best_params.json'
    
    with open(params_file, 'r') as f:
        data = json.load(f)
    return data['best_params']


def load_best_trial_info(params_file: str = None) -> dict:
    """
    Load full information about best trial including all metrics.
    
    Args:
        params_file: Path to best_params.json (default: auto-detect from current directory)
        
    Returns:
        dict with params, metrics, trial_name, and trial_path
    """
    if params_file is None:
        # Try to find the file relative to current working directory
        cwd = Path.cwd()
        # Check if we're in notebooks/ directory
        if cwd.name == 'notebooks':
            params_file = '../logs/optimization/best_params.json'
        else:
            params_file = 'logs/optimization/best_params.json'
    
    with open(params_file, 'r') as f:
        return json.load(f)


def print_best_params(params_file: str = None):
    """
    Pretty print best parameters and performance.
    
    Args:
        params_file: Path to best_params.json (default: auto-detect from current directory)
    """
    info = load_best_trial_info(params_file)
    
    print("=" * 60)
    print("OPTIMIZED PARAMETERS")
    print("=" * 60)
    print(f"Trial: {info['trial_name']}")
    print(f"Optimized for: {info['metric']}")
    print()
    
    print("Parameters:")
    for key, value in info['best_params'].items():
        if isinstance(value, float):
            print(f"  {key:20s} = {value:.4f}")
        else:
            print(f"  {key:20s} = {value}")
    
    print()
    print("Performance Metrics:")
    for key, value in info['all_metrics'].items():
        if isinstance(value, float):
            print(f"  {key:20s} = {value:.4f}")
        else:
            print(f"  {key:20s} = {value}")
    
    print("=" * 60)


if __name__ == '__main__':
    # Example usage
    print_best_params()
    
    print("\nTo use in your code:")
    print("=" * 60)
    print("""
from utils.load_params import load_best_params

# Load optimized parameters
params = load_best_params()

# Use in your signal
from signals.momentum import MomentumSignalV2

signal = MomentumSignalV2(
    lookback=params['lookback'],
    entry_threshold=params['entry_threshold'],
    exit_threshold=params['exit_threshold']
)

# Use in your backtest
from core.paper_trader import PaperTrader

trader = PaperTrader(initial_cash=100000)
results = trader.simulate(
    df=market_data,
    stop_loss_pct=params['stop_loss_pct']
)
""")
