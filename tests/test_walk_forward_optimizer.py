"""
Tests for WalkForwardOptimizer

Run with: pytest tests/test_walk_forward_optimizer.py -v
"""

import pytest
import pandas as pd
import numpy as np

from core.walk_forward_optimizer import WalkForwardOptimizer, WalkForwardPeriod
from signals.momentum import MomentumSignalV2


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    prices = pd.DataFrame({
        'Open': 4500 + np.cumsum(np.random.randn(500)) * 10,
        'High': 4520 + np.cumsum(np.random.randn(500)) * 10,
        'Low': 4480 + np.cumsum(np.random.randn(500)) * 10,
        'Close': 4500 + np.cumsum(np.random.randn(500)) * 10,
        'Volume': 1000000 + np.random.randint(-100000, 100000, 500)
    }, index=dates)
    
    return prices


class TestWalkForwardOptimizer:
    """Test walk-forward optimization."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = WalkForwardOptimizer(
            signal_class=MomentumSignalV2,
            param_grid={'lookback': [10, 20, 30]},
            train_pct=0.70,
            test_pct=0.15
        )
        
        assert optimizer.signal_class == MomentumSignalV2
        assert optimizer.train_pct == 0.70
        assert optimizer.test_pct == 0.15
        assert optimizer.metric == 'sharpe'
    
    def test_invalid_percentages(self):
        """Test that invalid percentages raise errors."""
        with pytest.raises(ValueError, match="train_pct must be between 0 and 1"):
            WalkForwardOptimizer(
                signal_class=MomentumSignalV2,
                param_grid={'lookback': [10]},
                train_pct=1.5
            )
        
        with pytest.raises(ValueError, match="train_pct \\+ test_pct must be <= 1"):
            WalkForwardOptimizer(
                signal_class=MomentumSignalV2,
                param_grid={'lookback': [10]},
                train_pct=0.70,
                test_pct=0.40
            )
    
    def test_param_combinations(self):
        """Test parameter combination generation."""
        optimizer = WalkForwardOptimizer(
            signal_class=MomentumSignalV2,
            param_grid={
                'lookback': [10, 20],
                'entry_threshold': [0.01, 0.02]
            },
            train_pct=0.70,
            test_pct=0.15
        )
        
        combinations = optimizer._generate_param_combinations()
        assert len(combinations) == 4  # 2 * 2
        assert {'lookback': 10, 'entry_threshold': 0.01} in combinations
        assert {'lookback': 20, 'entry_threshold': 0.02} in combinations
    
    def test_window_calculation(self):
        """Test train/test window calculation."""
        optimizer = WalkForwardOptimizer(
            signal_class=MomentumSignalV2,
            param_grid={'lookback': [20]},
            train_pct=0.70,
            test_pct=0.15,
            min_train_days=50,
            min_test_days=10
        )
        
        windows = optimizer._calculate_windows(total_days=300)
        
        # Should have at least one window
        assert len(windows) > 0
        
        # Check first window
        train_start, train_end, test_start, test_end = windows[0]
        assert train_start == 0
        assert train_end == int(300 * 0.70)  # 210
        assert test_start == train_end
        assert test_end <= 300
    
    def test_optimize_runs(self, sample_data):
        """Test that optimization completes successfully."""
        optimizer = WalkForwardOptimizer(
            signal_class=MomentumSignalV2,
            param_grid={'lookback': [20, 30]},
            train_pct=0.60,
            test_pct=0.20,
            min_train_days=100,
            min_test_days=30,
            verbose=False
        )
        
        periods = optimizer.optimize(
            prices=sample_data,
            asset_name='TEST',
            initial_capital=100000
        )
        
        # Should have at least one period
        assert len(periods) > 0
        assert all(isinstance(p, WalkForwardPeriod) for p in periods)
        
        # Check first period has required attributes
        period = periods[0]
        assert period.period_num == 1
        assert isinstance(period.best_params, dict)
        assert 'lookback' in period.best_params
        assert isinstance(period.test_return, float)
        assert isinstance(period.test_sharpe, float)
    
    def test_get_best_params(self, sample_data):
        """Test getting most stable parameters."""
        optimizer = WalkForwardOptimizer(
            signal_class=MomentumSignalV2,
            param_grid={'lookback': [20, 30, 40]},
            train_pct=0.60,
            test_pct=0.20,
            min_train_days=100,
            min_test_days=30,
            verbose=False
        )
        
        optimizer.optimize(sample_data, 'TEST', 100000)
        best_params = optimizer.get_best_params()
        
        assert isinstance(best_params, dict)
        assert 'lookback' in best_params
        assert best_params['lookback'] in [20, 30, 40]
    
    def test_summary_statistics(self, sample_data):
        """Test summary statistics calculation."""
        optimizer = WalkForwardOptimizer(
            signal_class=MomentumSignalV2,
            param_grid={'lookback': [20]},
            train_pct=0.60,
            test_pct=0.20,
            min_train_days=100,
            min_test_days=30,
            verbose=False
        )
        
        optimizer.optimize(sample_data, 'TEST', 100000)
        stats = optimizer.get_summary_statistics()
        
        assert 'num_periods' in stats
        assert 'avg_return' in stats
        assert 'win_rate' in stats
        assert 'total_return' in stats
        assert stats['num_periods'] > 0
        assert 0 <= stats['win_rate'] <= 1
    
    def test_parameter_stability_dataframe(self, sample_data):
        """Test parameter stability DataFrame creation."""
        optimizer = WalkForwardOptimizer(
            signal_class=MomentumSignalV2,
            param_grid={'lookback': [20, 30]},
            train_pct=0.60,
            test_pct=0.20,
            min_train_days=100,
            min_test_days=30,
            verbose=False
        )
        
        optimizer.optimize(sample_data, 'TEST', 100000)
        stability_df = optimizer.get_parameter_stability()
        
        assert isinstance(stability_df, pd.DataFrame)
        assert 'Period' in stability_df.columns
        assert 'lookback' in stability_df.columns
        assert len(stability_df) == len(optimizer.periods)
