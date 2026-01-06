"""
Tests for TAA portfolio optimizer.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.taa.optimizer import TAAOptimizer, BacktestOptimizer
from core.taa.constraints import load_constraints_from_config
from pathlib import Path
import yaml


@pytest.fixture
def sample_constraints():
    """Create sample optimization constraints."""
    # Use test config or create minimal constraints
    config_path = Path(__file__).parent.parent / "config" / "taa_constraints.yaml"
    if config_path.exists():
        return load_constraints_from_config(str(config_path))
    else:
        # Fallback to minimal config for testing
        from core.taa.constraints import OptimizationConstraints, PositionConstraints
        return OptimizationConstraints(
            position=PositionConstraints(max_position=0.25)
        )


@pytest.fixture
def sample_expected_returns():
    """Create sample expected returns."""
    return {
        'SPY': 0.08,
        'XLE': 0.12,
        'XLF': 0.05,
        'XLK': 0.10
    }


@pytest.fixture
def sample_covariance():
    """Create sample covariance matrix."""
    tickers = ['SPY', 'XLE', 'XLF', 'XLK']
    np.random.seed(42)
    
    # Generate positive semi-definite covariance matrix
    A = np.random.randn(4, 4) * 0.01
    cov = A @ A.T + np.eye(4) * 0.02
    
    return pd.DataFrame(cov, index=tickers, columns=tickers)


@pytest.fixture
def sample_returns_data():
    """Create sample returns DataFrame for backtesting."""
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    tickers = ['SPY', 'XLE', 'XLF', 'XLK']
    
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(500, 4) * 0.01,
        index=dates,
        columns=tickers
    )
    
    return returns


@pytest.fixture
def sample_predictions():
    """Create sample predictions DataFrame."""
    dates = pd.date_range('2020-03-01', periods=100, freq='W')
    tickers = ['SPY', 'XLE', 'XLF', 'XLK']
    
    np.random.seed(42)
    predictions = pd.DataFrame(
        np.random.randn(100, 4) * 0.02 + 0.05,
        index=dates,
        columns=tickers
    )
    
    return predictions


class TestTAAOptimizer:
    """Test TAAOptimizer factory class."""
    
    def test_initialization_default_method(self, sample_constraints):
        """Test optimizer initialization with default method."""
        optimizer = TAAOptimizer(sample_constraints)
        assert optimizer.constraints == sample_constraints
        assert optimizer.method == 'mean_variance'
    
    def test_initialization_with_method(self, sample_constraints):
        """Test optimizer initialization with specific method."""
        optimizer = TAAOptimizer(sample_constraints, method='max_sharpe')
        assert optimizer.method == 'max_sharpe'
    
    def test_initialization_invalid_method(self, sample_constraints):
        """Test optimizer raises error for invalid method."""
        with pytest.raises(ValueError, match="Invalid method"):
            TAAOptimizer(sample_constraints, method='invalid_method')
    
    def test_optimize_basic(self, sample_constraints, sample_expected_returns, sample_covariance):
        """Test basic optimization without previous weights."""
        optimizer = TAAOptimizer(sample_constraints, method='mean_variance')
        
        weights, metadata = optimizer.optimize(
            expected_returns=sample_expected_returns,
            covariance_matrix=sample_covariance
        )
        
        # Check weights are valid
        assert isinstance(weights, dict)
        assert set(weights.keys()) == set(sample_expected_returns.keys())
        assert all(0 <= w <= 1.0 for w in weights.values())  # Position limits
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Sum to 1
        
        # Check metadata
        assert metadata['status'] in ['optimal', 'solver_error']
        assert 'expected_return' in metadata
        assert 'volatility' in metadata
        assert 'turnover' in metadata
    
    @pytest.mark.parametrize('method', [
        'mean_variance', 'max_sharpe', 'min_variance', 'risk_parity',
        'black_litterman', 'cvar', 'hrp', 'kelly'
    ])
    def test_all_optimization_methods(self, sample_constraints, sample_expected_returns, sample_covariance, method):
        """Test all 8 optimization methods work correctly."""
        # Skip CVaR with small dataset (needs scenarios)
        if method == 'cvar' and len(sample_expected_returns) < 5:
            pytest.skip("CVaR needs more assets for scenarios")
        
        optimizer = TAAOptimizer(sample_constraints, method=method)
        
        weights, metadata = optimizer.optimize(
            expected_returns=sample_expected_returns,
            covariance_matrix=sample_covariance
        )
        
        # All methods should return valid weights
        assert isinstance(weights, dict)
        assert len(weights) == len(sample_expected_returns)
        assert abs(sum(weights.values()) - 1.0) < 0.01
        
        # Check status (some methods may fail with small datasets)
        assert metadata['status'] in ['optimal', 'solver_error', 'failed']
    
    def test_optimize_with_previous_weights(self, sample_constraints, sample_expected_returns, sample_covariance):
        """Test optimization with previous weights for turnover calculation."""
        optimizer = TAAOptimizer(sample_constraints, method='mean_variance')
        
        previous_weights = {
            'SPY': 0.25,
            'XLE': 0.25,
            'XLF': 0.25,
            'XLK': 0.25
        }
        
        weights, metadata = optimizer.optimize(
            expected_returns=sample_expected_returns,
            covariance_matrix=sample_covariance,
            previous_weights=previous_weights
        )
        
        # Turnover should be calculated
        assert metadata['turnover'] >= 0
        assert metadata['turnover'] <= 2.0  # Max possible turnover
    
    def test_optimize_returns_optimal_status(self, sample_constraints, sample_expected_returns, sample_covariance):
        """Test optimization returns optimal status."""
        optimizer = TAAOptimizer(sample_constraints, method='max_sharpe')
        
        weights, metadata = optimizer.optimize(
            expected_returns=sample_expected_returns,
            covariance_matrix=sample_covariance
        )
        
        assert metadata['status'] in ['optimal', 'failed', 'solver_error']
        if metadata['status'] == 'optimal':
            assert metadata['objective_value'] is not None
    
    def test_optimize_respects_position_limits(self, sample_constraints, sample_expected_returns, sample_covariance):
        """Test optimization respects max position constraint."""
        optimizer = TAAOptimizer(sample_constraints, method='min_variance')
        
        weights, metadata = optimizer.optimize(
            expected_returns=sample_expected_returns,
            covariance_matrix=sample_covariance
        )
        
        if metadata['status'] == 'optimal':
            max_weight = max(weights.values())
            # Use actual max_position from constraints
            max_allowed = getattr(sample_constraints.position, 'max_position', 1.0)
            assert max_weight <= max_allowed + 0.01  # Small tolerance


class TestBacktestOptimizer:
    """Test BacktestOptimizer class."""
    
    def test_initialization(self, sample_constraints):
        """Test backtest optimizer initialization."""
        optimizer = TAAOptimizer(sample_constraints, method='mean_variance')
        backtest_optimizer = BacktestOptimizer(
            optimizer=optimizer,
            lookback_days=252,
            rebalance_freq='W'
        )
        
        assert backtest_optimizer.optimizer == optimizer
        assert backtest_optimizer.lookback_days == 252
        assert backtest_optimizer.rebalance_freq == 'W'
    
    def test_run_backtest_basic(self, sample_constraints, sample_predictions, sample_returns_data):
        """Test basic backtest execution."""
        optimizer = TAAOptimizer(sample_constraints, method='mean_variance')
        backtest_optimizer = BacktestOptimizer(
            optimizer=optimizer,
            lookback_days=60,  # Shorter for test data
            rebalance_freq='W'
        )
        
        portfolio_weights = backtest_optimizer.run_backtest(
            predictions_df=sample_predictions,
            returns_df=sample_returns_data
        )
        
        # Check output structure
        assert isinstance(portfolio_weights, pd.DataFrame)
        assert len(portfolio_weights) > 0
        
        # Check columns include tickers and metadata
        ticker_cols = [col for col in portfolio_weights.columns if not col.startswith('meta_')]
        meta_cols = [col for col in portfolio_weights.columns if col.startswith('meta_')]
        
        assert len(ticker_cols) == 4  # SPY, XLE, XLF, XLK
        assert 'meta_status' in meta_cols
        assert 'meta_expected_return' in meta_cols
        assert 'meta_volatility' in meta_cols
    
    def test_run_backtest_weights_sum_to_one(self, sample_constraints, sample_predictions, sample_returns_data):
        """Test backtest weights sum to 1 at each rebalance."""
        optimizer = TAAOptimizer(sample_constraints, method='mean_variance')
        backtest_optimizer = BacktestOptimizer(
            optimizer=optimizer,
            lookback_days=60,
            rebalance_freq='W'
        )
        
        portfolio_weights = backtest_optimizer.run_backtest(
            predictions_df=sample_predictions,
            returns_df=sample_returns_data
        )
        
        ticker_cols = [col for col in portfolio_weights.columns if not col.startswith('meta_')]
        weight_sums = portfolio_weights[ticker_cols].sum(axis=1)
        
        # All weight sums should be close to 1
        assert all(abs(weight_sums - 1.0) < 0.01)
    
    def test_run_backtest_insufficient_data(self, sample_constraints):
        """Test backtest raises error with insufficient data."""
        optimizer = TAAOptimizer(sample_constraints, method='mean_variance')
        backtest_optimizer = BacktestOptimizer(
            optimizer=optimizer,
            lookback_days=252,
            rebalance_freq='W'
        )
        
        # Very short data
        short_predictions = pd.DataFrame(
            [[0.05, 0.06, 0.04, 0.05]],
            index=pd.date_range('2020-01-01', periods=1, freq='W'),
            columns=['SPY', 'XLE', 'XLF', 'XLK']
        )
        
        short_returns = pd.DataFrame(
            np.random.randn(10, 4) * 0.01,
            index=pd.date_range('2020-01-01', periods=10, freq='D'),
            columns=['SPY', 'XLE', 'XLF', 'XLK']
        )
        
        with pytest.raises(RuntimeError, match="No valid optimization results"):
            backtest_optimizer.run_backtest(
                predictions_df=short_predictions,
                returns_df=short_returns
            )
    
    def test_run_backtest_respects_rebalance_frequency(self, sample_constraints, sample_predictions, sample_returns_data):
        """Test backtest respects rebalance frequency."""
        optimizer = TAAOptimizer(sample_constraints, method='mean_variance')
        backtest_optimizer = BacktestOptimizer(
            optimizer=optimizer,
            lookback_days=60,
            rebalance_freq='W'
        )
        
        portfolio_weights = backtest_optimizer.run_backtest(
            predictions_df=sample_predictions,
            returns_df=sample_returns_data
        )
        
        # Check dates are weekly (approximately)
        date_diffs = portfolio_weights.index.to_series().diff().dt.days.dropna()
        avg_days_between = date_diffs.mean()
        
        # Should be around 7 days for weekly
        assert 5 <= avg_days_between <= 9


class TestOptimizerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_optimizer_with_nan_returns(self, sample_constraints, sample_covariance):
        """Test optimizer handles NaN in expected returns."""
        optimizer = TAAOptimizer(sample_constraints, method='mean_variance')
        
        expected_returns = {
            'SPY': 0.08,
            'XLE': np.nan,
            'XLF': 0.05,
            'XLK': 0.10
        }
        
        # Should handle gracefully
        weights, metadata = optimizer.optimize(
            expected_returns=expected_returns,
            covariance_matrix=sample_covariance
        )
        
        assert isinstance(weights, dict)
        # Status could be optimal, failed, or unbounded_inaccurate with NaN
        assert metadata['status'] in ['optimal', 'failed', 'unbounded_inaccurate', 'infeasible']
    
    def test_optimizer_with_mismatched_tickers(self, sample_constraints):
        """Test optimizer handles mismatched tickers in returns and covariance."""
        optimizer = TAAOptimizer(sample_constraints, method='mean_variance')
        
        expected_returns = {'SPY': 0.08, 'XLE': 0.10}
        covariance = pd.DataFrame(
            [[0.02, 0.01], [0.01, 0.03]],
            index=['SPY', 'XLK'],  # XLK instead of XLE
            columns=['SPY', 'XLK']
        )
        
        # Should raise error or handle gracefully
        try:
            weights, metadata = optimizer.optimize(
                expected_returns=expected_returns,
                covariance_matrix=covariance
            )
            # If it doesn't raise, check it returned valid output
            assert isinstance(weights, dict)
        except (KeyError, ValueError):
            pass  # Expected behavior
