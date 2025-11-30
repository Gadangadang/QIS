"""
Parameter Optimization Framework
Supports grid search and random search for strategy parameter tuning.
"""
import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, List, Any, Callable, Tuple
import json
from pathlib import Path


class ParameterOptimizer:
    """
    Optimizes strategy parameters using grid search or random search.
    
    Usage:
        optimizer = ParameterOptimizer(
            objective_fn=run_backtest_with_params,
            param_grid={'lookback': [50, 100, 200], 'stop_loss': [0.05, 0.10]},
            metric='sharpe'
        )
        best_params, best_score, results = optimizer.grid_search()
    """
    
    def __init__(
        self,
        objective_fn: Callable,
        param_grid: Dict[str, List],
        metric: str = 'sharpe',
        maximize: bool = True,
        verbose: bool = True
    ):
        """
        Initialize optimizer.
        
        Args:
            objective_fn: Function that takes params dict and returns results dict with metrics
            param_grid: Dict mapping parameter names to list of values to try
            metric: Metric to optimize ('sharpe', 'total_return', 'calmar', 'win_rate', etc.)
            maximize: True to maximize metric, False to minimize
            verbose: Print progress during optimization
        """
        self.objective_fn = objective_fn
        self.param_grid = param_grid
        self.metric = metric
        self.maximize = maximize
        self.verbose = verbose
        
    def grid_search(self) -> Tuple[Dict[str, Any], float, pd.DataFrame]:
        """
        Exhaustive grid search over all parameter combinations.
        
        Returns:
            best_params: Dict of best parameter values
            best_score: Best metric value achieved
            results_df: DataFrame with all trials and scores
        """
        # Generate all combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(product(*param_values))
        
        if self.verbose:
            print(f"Grid Search: Testing {len(combinations)} combinations")
            print(f"Parameters: {param_names}")
            print(f"Optimizing: {self.metric} ({'maximize' if self.maximize else 'minimize'})")
            print("=" * 60)
        
        results = []
        best_score = -np.inf if self.maximize else np.inf
        best_params = None
        
        for i, combo in enumerate(combinations, 1):
            # Build params dict
            params = dict(zip(param_names, combo))
            
            try:
                # Run backtest with these params
                result = self.objective_fn(params)
                score = result.get(self.metric, np.nan)
                
                # Track results
                trial_result = {**params, self.metric: score}
                results.append(trial_result)
                
                # Update best
                if not np.isnan(score):
                    if (self.maximize and score > best_score) or \
                       (not self.maximize and score < best_score):
                        best_score = score
                        best_params = params.copy()
                
                if self.verbose:
                    print(f"Trial {i}/{len(combinations)}: {params} -> {self.metric}={score:.4f}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Trial {i}/{len(combinations)}: {params} -> ERROR: {str(e)}")
                results.append({**params, self.metric: np.nan, 'error': str(e)})
        
        results_df = pd.DataFrame(results)
        
        if self.verbose:
            print("=" * 60)
            print(f"Best {self.metric}: {best_score:.4f}")
            print(f"Best params: {best_params}")
        
        return best_params, best_score, results_df
    
    def random_search(
        self, 
        n_iter: int = 50,
        random_state: int = None
    ) -> Tuple[Dict[str, Any], float, pd.DataFrame]:
        """
        Random search over parameter space.
        
        Args:
            n_iter: Number of random combinations to try
            random_state: Random seed for reproducibility
            
        Returns:
            best_params: Dict of best parameter values
            best_score: Best metric value achieved
            results_df: DataFrame with all trials and scores
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        param_names = list(self.param_grid.keys())
        
        if self.verbose:
            print(f"Random Search: Testing {n_iter} random combinations")
            print(f"Parameters: {param_names}")
            print(f"Optimizing: {self.metric} ({'maximize' if self.maximize else 'minimize'})")
            print("=" * 60)
        
        results = []
        best_score = -np.inf if self.maximize else np.inf
        best_params = None
        
        for i in range(n_iter):
            # Randomly sample parameters
            params = {
                name: np.random.choice(values)
                for name, values in self.param_grid.items()
            }
            
            try:
                # Run backtest
                result = self.objective_fn(params)
                score = result.get(self.metric, np.nan)
                
                # Track results
                trial_result = {**params, self.metric: score}
                results.append(trial_result)
                
                # Update best
                if not np.isnan(score):
                    if (self.maximize and score > best_score) or \
                       (not self.maximize and score < best_score):
                        best_score = score
                        best_params = params.copy()
                
                if self.verbose:
                    print(f"Trial {i+1}/{n_iter}: {params} -> {self.metric}={score:.4f}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Trial {i+1}/{n_iter}: {params} -> ERROR: {str(e)}")
                results.append({**params, self.metric: np.nan, 'error': str(e)})
        
        results_df = pd.DataFrame(results)
        
        if self.verbose:
            print("=" * 60)
            print(f"Best {self.metric}: {best_score:.4f}")
            print(f"Best params: {best_params}")
        
        return best_params, best_score, results_df


def save_optimization_results(
    best_params: Dict,
    best_score: float,
    results_df: pd.DataFrame,
    output_dir: str,
    fold_idx: int = None
):
    """
    Save optimization results to disk.
    
    Args:
        best_params: Best parameter dict
        best_score: Best score achieved
        results_df: DataFrame with all trials
        output_dir: Directory to save results
        fold_idx: Optional fold index for walk-forward optimization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save best params as JSON
    suffix = f"_fold_{fold_idx}" if fold_idx is not None else ""
    params_file = output_path / f"best_params{suffix}.json"
    
    with open(params_file, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': best_score
        }, f, indent=2)
    
    # Save all results as CSV
    results_file = output_path / f"optimization_results{suffix}.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"Saved optimization results to {output_dir}/")


def load_best_params(params_file: str) -> Dict:
    """
    Load best parameters from JSON file.
    
    Args:
        params_file: Path to best_params.json
        
    Returns:
        Dict of best parameters
    """
    with open(params_file, 'r') as f:
        data = json.load(f)
    return data['best_params']
