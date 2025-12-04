"""
Walk-Forward Optimization for Backtesting

This module provides walk-forward validation with parameter optimization to ensure
strategies are robust across different market regimes.

Key Features:
- Percentage-based train/test splits (e.g., 80% train, 20% test)
- Parameter grid search optimization
- Multiple optimization metrics (Sharpe, Total Return, Risk-Adjusted Return)
- Rolling window approach
- Out-of-sample validation on each test period

Example Usage:
    optimizer = WalkForwardOptimizer(
        signal_class=MomentumSignalV2,
        param_grid={'lookback': [10, 20, 30, 50]},
        train_pct=0.70,
        test_pct=0.15,
        metric='sharpe'
    )
    
    results = optimizer.optimize(prices, initial_capital=100000)
    best_params = optimizer.get_best_params()
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from itertools import product
from tqdm import tqdm

from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2


@dataclass
class WalkForwardPeriod:
    """Results from a single walk-forward period."""
    period_num: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: Dict[str, Any]
    test_return: float
    test_sharpe: float
    test_max_dd: float
    optimization_metric: float
    equity_curve: pd.DataFrame


@dataclass
class OptimizationResult:
    """Results from parameter optimization in train period."""
    params: Dict[str, Any]
    metric_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float


class WalkForwardOptimizer:
    """
    Walk-forward optimization with parameter tuning.
    
    This class implements a rolling window walk-forward validation approach:
    1. Split data into overlapping train/test windows (percentage-based)
    2. For each window:
       a. Optimize parameters on train set using grid search
       b. Test best parameters on out-of-sample test set
    3. Aggregate results across all periods
    
    Attributes:
        signal_class: Signal generator class (not instance)
        param_grid: Dictionary of parameter names to lists of values
        train_pct: Percentage of data for training (e.g., 0.70 = 70%)
        test_pct: Percentage of data for testing (e.g., 0.15 = 15%)
        metric: Optimization metric ('sharpe', 'return', 'risk_adjusted')
        min_train_days: Minimum days required in train period
        min_test_days: Minimum days required in test period
    """
    
    def __init__(
        self,
        signal_class,
        param_grid: Dict[str, List[Any]],
        train_pct: float = 0.70,
        test_pct: float = 0.15,
        metric: str = 'sharpe',
        min_train_days: int = 252,
        min_test_days: int = 60,
        risk_per_trade: float = 0.02,
        max_position_size: float = 1.0,
        transaction_cost_bps: float = 3.0,
        slippage_bps: float = 1.0,
        verbose: bool = True
    ):
        """
        Initialize walk-forward optimizer.
        
        Args:
            signal_class: Signal generator class (e.g., MomentumSignalV2)
            param_grid: Dict of param names to lists of values
                       e.g., {'lookback': [10, 20, 30], 'threshold': [0.01, 0.02]}
            train_pct: Training window size as percentage of total data
            test_pct: Test window size as percentage of total data
            metric: Optimization metric ('sharpe', 'return', 'risk_adjusted')
            min_train_days: Minimum trading days required in train period
            min_test_days: Minimum trading days required in test period
            risk_per_trade: Risk per trade for portfolio manager
            max_position_size: Maximum position size
            transaction_cost_bps: Transaction costs in basis points
            slippage_bps: Slippage in basis points
            verbose: Print progress information
        """
        self.signal_class = signal_class
        self.param_grid = param_grid
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.metric = metric.lower()
        self.min_train_days = min_train_days
        self.min_test_days = min_test_days
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.verbose = verbose
        
        # Results storage
        self.periods: List[WalkForwardPeriod] = []
        self.all_optimization_results: List[OptimizationResult] = []
        
        # Validate inputs
        if not (0 < train_pct < 1):
            raise ValueError(f"train_pct must be between 0 and 1, got {train_pct}")
        if not (0 < test_pct < 1):
            raise ValueError(f"test_pct must be between 0 and 1, got {test_pct}")
        if train_pct + test_pct > 1:
            raise ValueError(f"train_pct + test_pct must be <= 1, got {train_pct + test_pct}")
        if self.metric not in ['sharpe', 'return', 'risk_adjusted']:
            raise ValueError(f"metric must be 'sharpe', 'return', or 'risk_adjusted', got {metric}")
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _optimize_parameters(
        self,
        prices: pd.DataFrame,
        asset_name: str,
        initial_capital: float
    ) -> OptimizationResult:
        """
        Optimize parameters on training data using grid search.
        
        Args:
            prices: Training period price data
            asset_name: Name of the asset
            initial_capital: Starting capital
            
        Returns:
            OptimizationResult with best parameters and metrics
        """
        param_combinations = self._generate_param_combinations()
        best_result = None
        best_metric_value = -np.inf
        
        for params in param_combinations:
            try:
                # Create signal generator with these parameters
                signal_gen = self.signal_class(**params)
                
                # Generate signals
                signals = signal_gen.generate(prices.copy())
                
                # Run backtest
                pm = PortfolioManagerV2(
                    initial_capital=initial_capital,
                    risk_per_trade=self.risk_per_trade,
                    max_position_size=self.max_position_size,
                    transaction_cost_bps=self.transaction_cost_bps,
                    slippage_bps=self.slippage_bps
                )
                
                result = pm.run_backtest(
                    signals={asset_name: signals},
                    prices={asset_name: prices}
                )
                
                # Extract metrics
                total_return = result.metrics['Total Return']
                sharpe = result.metrics['Sharpe Ratio']
                max_dd = result.metrics['Max Drawdown']
                
                # Calculate optimization metric
                if self.metric == 'sharpe':
                    metric_value = sharpe
                elif self.metric == 'return':
                    metric_value = total_return
                else:  # risk_adjusted
                    metric_value = total_return / abs(max_dd) if max_dd != 0 else 0
                
                # Track best
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_result = OptimizationResult(
                        params=params,
                        metric_value=metric_value,
                        total_return=total_return,
                        sharpe_ratio=sharpe,
                        max_drawdown=max_dd
                    )
            
            except Exception as e:
                if self.verbose:
                    print(f"   Warning: Failed to evaluate params {params}: {e}")
                continue
        
        if best_result is None:
            raise ValueError("No valid parameter combinations found")
        
        return best_result
    
    def _calculate_windows(
        self,
        total_days: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Calculate train/test window indices using percentage-based approach.
        
        Args:
            total_days: Total number of trading days
            
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        train_size = int(total_days * self.train_pct)
        test_size = int(total_days * self.test_pct)
        step_size = test_size  # Roll forward by test period size
        
        windows = []
        current_start = 0
        
        while True:
            train_start = current_start
            train_end = train_start + train_size
            test_start = train_end
            test_end = test_start + test_size
            
            # Check if we have enough data
            if train_end - train_start < self.min_train_days:
                break
            if test_end > total_days:
                # Last window - test to end
                test_end = total_days
                if test_end - test_start < self.min_test_days:
                    break
            
            windows.append((train_start, train_end, test_start, test_end))
            
            # If this was the last window, stop
            if test_end >= total_days:
                break
            
            # Roll forward
            current_start += step_size
        
        return windows
    
    def optimize(
        self,
        prices: pd.DataFrame,
        asset_name: str = 'Asset',
        initial_capital: float = 100000
    ) -> List[WalkForwardPeriod]:
        """
        Run walk-forward optimization on price data.
        
        Args:
            prices: DataFrame with OHLC data
            asset_name: Name of the asset being tested
            initial_capital: Starting capital for each period
            
        Returns:
            List of WalkForwardPeriod results
        """
        if self.verbose:
            print("="*80)
            print("🔄 WALK-FORWARD OPTIMIZATION")
            print("="*80)
            print(f"Signal Class: {self.signal_class.__name__}")
            print(f"Parameter Grid: {self.param_grid}")
            print(f"Train/Test Split: {self.train_pct:.0%} / {self.test_pct:.0%}")
            print(f"Optimization Metric: {self.metric.upper()}")
            print(f"Data: {len(prices)} days ({prices.index[0].date()} to {prices.index[-1].date()})")
            print("="*80)
        
        # Calculate windows
        windows = self._calculate_windows(len(prices))
        
        if self.verbose:
            print(f"\n📊 Generated {len(windows)} walk-forward periods\n")
        
        # Run optimization for each period
        self.periods = []
        
        iterator = enumerate(windows, 1)
        if self.verbose:
            iterator = tqdm(list(iterator), desc="Walk-Forward Periods")
        
        for period_num, (train_start, train_end, test_start, test_end) in iterator:
            # Get train/test data
            train_prices = prices.iloc[train_start:train_end].copy()
            test_prices = prices.iloc[test_start:test_end].copy()
            
            if self.verbose and not isinstance(iterator, tqdm):
                print(f"\nPeriod {period_num}:")
                print(f"  Train: {train_prices.index[0].date()} to {train_prices.index[-1].date()} ({len(train_prices)} days)")
                print(f"  Test:  {test_prices.index[0].date()} to {test_prices.index[-1].date()} ({len(test_prices)} days)")
            
            # Optimize on train data
            if self.verbose and not isinstance(iterator, tqdm):
                print(f"  Optimizing {len(self._generate_param_combinations())} parameter combinations...")
            
            opt_result = self._optimize_parameters(train_prices, asset_name, initial_capital)
            
            if self.verbose and not isinstance(iterator, tqdm):
                print(f"  Best params: {opt_result.params}")
                print(f"  Train {self.metric}: {opt_result.metric_value:.4f}")
            
            # Test on out-of-sample data with best parameters
            signal_gen = self.signal_class(**opt_result.params)
            test_signals = signal_gen.generate(test_prices.copy())
            
            pm = PortfolioManagerV2(
                initial_capital=initial_capital,
                risk_per_trade=self.risk_per_trade,
                max_position_size=self.max_position_size,
                transaction_cost_bps=self.transaction_cost_bps,
                slippage_bps=self.slippage_bps
            )
            
            test_result = pm.run_backtest(
                signals={asset_name: test_signals},
                prices={asset_name: test_prices}
            )
            
            # Store period results
            period = WalkForwardPeriod(
                period_num=period_num,
                train_start=train_prices.index[0],
                train_end=train_prices.index[-1],
                test_start=test_prices.index[0],
                test_end=test_prices.index[-1],
                best_params=opt_result.params,
                test_return=test_result.metrics['Total Return'],
                test_sharpe=test_result.metrics['Sharpe Ratio'],
                test_max_dd=test_result.metrics['Max Drawdown'],
                optimization_metric=opt_result.metric_value,
                equity_curve=test_result.equity_curve
            )
            
            self.periods.append(period)
            self.all_optimization_results.append(opt_result)
            
            if self.verbose and not isinstance(iterator, tqdm):
                print(f"  Test Return: {period.test_return:.2%}")
                print(f"  Test Sharpe: {period.test_sharpe:.4f}")
                print(f"  Test Max DD: {period.test_max_dd:.2%}")
        
        if self.verbose:
            self._print_summary()
        
        return self.periods
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the most frequently selected parameters across all periods.
        
        Returns:
            Dictionary of parameter names to most common values
        """
        if not self.periods:
            raise ValueError("Must run optimize() before calling get_best_params()")
        
        # Count frequency of each parameter value
        param_counts = {}
        for param_name in self.param_grid.keys():
            param_counts[param_name] = {}
        
        for period in self.periods:
            for param_name, param_value in period.best_params.items():
                if param_value not in param_counts[param_name]:
                    param_counts[param_name][param_value] = 0
                param_counts[param_name][param_value] += 1
        
        # Get most common value for each parameter
        best_params = {}
        for param_name, value_counts in param_counts.items():
            best_params[param_name] = max(value_counts.items(), key=lambda x: x[1])[0]
        
        return best_params
    
    def get_parameter_stability(self) -> pd.DataFrame:
        """
        Analyze how stable parameters are across periods.
        
        Returns:
            DataFrame with parameter values by period
        """
        if not self.periods:
            raise ValueError("Must run optimize() before calling get_parameter_stability()")
        
        data = []
        for period in self.periods:
            row = {'Period': period.period_num}
            row.update(period.best_params)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all walk-forward periods.
        
        Returns:
            Dictionary with aggregate metrics
        """
        if not self.periods:
            raise ValueError("Must run optimize() before calling get_summary_statistics()")
        
        returns = [p.test_return for p in self.periods]
        sharpes = [p.test_sharpe for p in self.periods]
        drawdowns = [p.test_max_dd for p in self.periods]
        
        return {
            'num_periods': len(self.periods),
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'best_return': np.max(returns),
            'worst_return': np.min(returns),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns),
            'avg_sharpe': np.mean(sharpes),
            'avg_max_drawdown': np.mean(drawdowns),
            'total_return': np.prod([1 + r for r in returns]) - 1,
        }
    
    def _print_summary(self):
        """Print summary of walk-forward results."""
        stats = self.get_summary_statistics()
        best_params = self.get_best_params()
        
        print("\n" + "="*80)
        print("📊 WALK-FORWARD SUMMARY")
        print("="*80)
        print(f"Total Periods:        {stats['num_periods']}")
        print(f"Average Return:       {stats['avg_return']:.2%}")
        print(f"Median Return:        {stats['median_return']:.2%}")
        print(f"Best Period:          {stats['best_return']:.2%}")
        print(f"Worst Period:         {stats['worst_return']:.2%}")
        print(f"Win Rate:             {stats['win_rate']:.1%}")
        print(f"Average Sharpe:       {stats['avg_sharpe']:.4f}")
        print(f"Average Max DD:       {stats['avg_max_drawdown']:.2%}")
        print(f"Cumulative Return:    {stats['total_return']:.2%}")
        print("\n" + "="*80)
        print("🎯 MOST STABLE PARAMETERS")
        print("="*80)
        for param_name, param_value in best_params.items():
            print(f"{param_name}: {param_value}")
        print("="*80)
