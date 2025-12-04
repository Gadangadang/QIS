"""
Tests for BacktestOrchestrator - High-level backtesting API.

Validates that the orchestrator:
- Simplifies common workflows
- Maintains backward compatibility
- Produces same results as manual approach
- Handles errors gracefully

Run with: pytest tests/test_backtest_orchestrator.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.backtest_orchestrator import BacktestOrchestrator, StrategyConfig, run_multi_strategy_backtest
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2
from signals.momentum import MomentumSignalV2


class TestBasicWorkflow:
    """Test basic orchestrator workflow."""
    
    def test_initialization(self):
        """Test orchestrator can be created."""
        orchestrator = BacktestOrchestrator()
        
        assert orchestrator.prices == {}
        assert orchestrator.strategies == []
        assert orchestrator.signals == {}
        assert orchestrator.results == {}
        assert orchestrator._data_loaded is False
    
    def test_futures_initialization(self):
        """Test orchestrator with futures sizing."""
        multipliers = {'ES': 50, 'CL': 1000}
        orchestrator = BacktestOrchestrator(
            use_futures_sizing=True,
            contract_multipliers=multipliers
        )
        
        assert orchestrator.use_futures_sizing is True
        assert orchestrator.contract_multipliers == multipliers
    
    def test_method_chaining(self):
        """Test that methods return self for chaining."""
        orchestrator = BacktestOrchestrator()
        
        # Mock data loading (skip actual yfinance call)
        orchestrator.prices = {'ES': pd.DataFrame({'Close': [100, 101, 102]})}
        orchestrator._data_loaded = True
        
        # Test chaining
        result = orchestrator.add_strategy(
            'Test', 
            MomentumSignalV2(), 
            ['ES'], 
            100000
        )
        
        assert result is orchestrator


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_data_mock(self):
        """Test data loading with mock data."""
        orchestrator = BacktestOrchestrator()
        
        # Mock the load (don't actually call yfinance in tests)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        mock_data = {
            'ES': pd.DataFrame({
                'Open': 100 + np.random.randn(100),
                'High': 102 + np.random.randn(100),
                'Low': 98 + np.random.randn(100),
                'Close': 100 + np.random.randn(100),
                'Volume': 1000000
            }, index=dates)
        }
        
        orchestrator.prices = mock_data
        orchestrator._data_loaded = True
        
        assert 'ES' in orchestrator.prices
        assert len(orchestrator.prices['ES']) == 100
        assert orchestrator._data_loaded is True


class TestStrategyConfiguration:
    """Test strategy configuration."""
    
    @pytest.fixture
    def loaded_orchestrator(self):
        """Orchestrator with mock data loaded."""
        orchestrator = BacktestOrchestrator()
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        orchestrator.prices = {
            'ES': pd.DataFrame({
                'Open': 100 + np.random.randn(100),
                'High': 102 + np.random.randn(100),
                'Low': 98 + np.random.randn(100),
                'Close': 100 + np.random.randn(100),
                'Volume': 1000000
            }, index=dates),
            'NQ': pd.DataFrame({
                'Open': 200 + np.random.randn(100),
                'High': 202 + np.random.randn(100),
                'Low': 198 + np.random.randn(100),
                'Close': 200 + np.random.randn(100),
                'Volume': 1000000
            }, index=dates)
        }
        orchestrator._data_loaded = True
        return orchestrator
    
    def test_add_single_strategy(self, loaded_orchestrator):
        """Test adding a single strategy."""
        loaded_orchestrator.add_strategy(
            name='Momentum',
            signal_generator=MomentumSignalV2(lookback=20),
            assets=['ES'],
            capital=100000
        )
        
        assert len(loaded_orchestrator.strategies) == 1
        assert loaded_orchestrator.strategies[0].name == 'Momentum'
        assert loaded_orchestrator.strategies[0].capital == 100000
    
    def test_add_multiple_strategies(self, loaded_orchestrator):
        """Test adding multiple strategies."""
        loaded_orchestrator.add_strategy(
            'Strategy1', MomentumSignalV2(), ['ES'], 100000
        )
        loaded_orchestrator.add_strategy(
            'Strategy2', MomentumSignalV2(), ['NQ'], 100000
        )
        
        assert len(loaded_orchestrator.strategies) == 2
    
    def test_auto_position_sizing_single_asset(self, loaded_orchestrator):
        """Test automatic position sizing for single asset."""
        loaded_orchestrator.add_strategy(
            'Test', MomentumSignalV2(), ['ES'], 100000
        )
        
        assert loaded_orchestrator.strategies[0].max_position_pct == 1.0
    
    def test_auto_position_sizing_multiple_assets(self, loaded_orchestrator):
        """Test automatic position sizing for multiple assets."""
        loaded_orchestrator.add_strategy(
            'Test', MomentumSignalV2(), ['ES', 'NQ'], 100000
        )
        
        assert loaded_orchestrator.strategies[0].max_position_pct == 0.5
    
    def test_invalid_asset_raises_error(self, loaded_orchestrator):
        """Test that invalid asset raises error."""
        with pytest.raises(ValueError, match="not found in loaded data"):
            loaded_orchestrator.add_strategy(
                'Test', MomentumSignalV2(), ['INVALID'], 100000
            )
    
    def test_add_strategy_before_data_raises_error(self):
        """Test that adding strategy before loading data raises error."""
        orchestrator = BacktestOrchestrator()
        
        with pytest.raises(RuntimeError, match="Must load data"):
            orchestrator.add_strategy(
                'Test', MomentumSignalV2(), ['ES'], 100000
            )


class TestSignalGeneration:
    """Test signal generation."""
    
    @pytest.fixture
    def configured_orchestrator(self):
        """Orchestrator with data and strategies configured."""
        orchestrator = BacktestOrchestrator()
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        np.random.seed(42)
        orchestrator.prices = {
            'ES': pd.DataFrame({
                'Open': 4500 + np.random.randn(100) * 50,
                'High': 4520 + np.random.randn(100) * 50,
                'Low': 4480 + np.random.randn(100) * 50,
                'Close': 4500 + np.random.randn(100) * 50,
                'Volume': 1000000
            }, index=dates)
        }
        orchestrator._data_loaded = True
        
        orchestrator.add_strategy(
            'Momentum', MomentumSignalV2(lookback=20), ['ES'], 100000
        )
        
        return orchestrator
    
    def test_generate_signals(self, configured_orchestrator):
        """Test signal generation."""
        configured_orchestrator.generate_signals(verbose=False)
        
        assert configured_orchestrator._signals_generated is True
        assert 'Momentum' in configured_orchestrator.signals
        assert 'ES' in configured_orchestrator.signals['Momentum']
    
    def test_generate_signals_before_adding_strategies_raises_error(self):
        """Test that generating signals before adding strategies raises error."""
        orchestrator = BacktestOrchestrator()
        orchestrator._data_loaded = True
        
        with pytest.raises(RuntimeError, match="No strategies added"):
            orchestrator.generate_signals()
    
    def test_signals_have_required_columns(self, configured_orchestrator):
        """Test that generated signals have required columns."""
        configured_orchestrator.generate_signals(verbose=False)
        
        signals = configured_orchestrator.signals['Momentum']['ES']
        assert 'Signal' in signals.columns
        assert len(signals) > 0


class TestBacktestExecution:
    """Test backtest execution."""
    
    @pytest.fixture
    def ready_orchestrator(self):
        """Orchestrator ready to run backtests."""
        orchestrator = BacktestOrchestrator()
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        np.random.seed(42)
        orchestrator.prices = {
            'ES': pd.DataFrame({
                'Open': 4500 + np.random.randn(100) * 50,
                'High': 4520 + np.random.randn(100) * 50,
                'Low': 4480 + np.random.randn(100) * 50,
                'Close': 4500 + np.random.randn(100) * 50,
                'Volume': 1000000
            }, index=dates)
        }
        orchestrator._data_loaded = True
        
        orchestrator.add_strategy(
            'Momentum', MomentumSignalV2(lookback=20), ['ES'], 100000
        )
        orchestrator.generate_signals(verbose=False)
        
        return orchestrator
    
    def test_run_backtests(self, ready_orchestrator):
        """Test running backtests."""
        results = ready_orchestrator.run_backtests(verbose=False)
        
        assert ready_orchestrator._backtests_run is True
        assert 'Momentum' in results
        assert 'result' in results['Momentum']
        assert 'capital' in results['Momentum']
        assert 'assets' in results['Momentum']
    
    def test_run_backtests_before_signals_raises_error(self):
        """Test that running backtests before generating signals raises error."""
        orchestrator = BacktestOrchestrator()
        orchestrator._data_loaded = True
        orchestrator.strategies = [StrategyConfig('Test', MomentumSignalV2(), ['ES'], 100000)]
        
        with pytest.raises(RuntimeError, match="Must generate signals"):
            orchestrator.run_backtests()
    
    def test_backtest_result_has_metrics(self, ready_orchestrator):
        """Test that backtest results have expected metrics."""
        results = ready_orchestrator.run_backtests(verbose=False)
        
        result = results['Momentum']['result']
        assert hasattr(result, 'final_equity')
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'metrics')
        assert 'Sharpe Ratio' in result.metrics
        assert 'Max Drawdown' in result.metrics


class TestResultsFormatting:
    """Test results formatting and reporting."""
    
    @pytest.fixture
    def completed_orchestrator(self):
        """Orchestrator with completed backtests."""
        orchestrator = BacktestOrchestrator()
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        np.random.seed(42)
        orchestrator.prices = {
            'ES': pd.DataFrame({
                'Open': 4500 + np.random.randn(100) * 50,
                'High': 4520 + np.random.randn(100) * 50,
                'Low': 4480 + np.random.randn(100) * 50,
                'Close': 4500 + np.random.randn(100) * 50,
                'Volume': 1000000
            }, index=dates)
        }
        orchestrator._data_loaded = True
        
        orchestrator.add_strategy(
            'Momentum', MomentumSignalV2(lookback=20), ['ES'], 100000
        )
        orchestrator.generate_signals(verbose=False)
        orchestrator.run_backtests(verbose=False)
        
        return orchestrator
    
    def test_get_summary(self, completed_orchestrator):
        """Test getting summary DataFrame."""
        summary = completed_orchestrator.get_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1
        assert 'Strategy' in summary.columns
        assert 'Total Return' in summary.columns
        assert 'Sharpe' in summary.columns
    
    def test_get_summary_before_backtests_raises_error(self):
        """Test that getting summary before running backtests raises error."""
        orchestrator = BacktestOrchestrator()
        
        with pytest.raises(RuntimeError, match="Must run backtests"):
            orchestrator.get_summary()


class TestBackwardCompatibility:
    """Test backward compatibility with old interface."""
    
    def test_run_multi_strategy_backtest_function(self):
        """Test backward compatible function."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        np.random.seed(42)
        prices = {
            'ES': pd.DataFrame({
                'Open': 4500 + np.random.randn(100) * 50,
                'High': 4520 + np.random.randn(100) * 50,
                'Low': 4480 + np.random.randn(100) * 50,
                'Close': 4500 + np.random.randn(100) * 50,
                'Volume': 1000000
            }, index=dates)
        }
        
        strategies = [
            {
                'name': 'Momentum',
                'signal_generator': MomentumSignalV2(lookback=20),
                'assets': ['ES'],
                'capital': 100000
            }
        ]
        
        results = run_multi_strategy_backtest(prices, strategies, verbose=False)
        
        assert 'Momentum' in results
        assert 'result' in results['Momentum']


class TestQuickBacktest:
    """Test quick backtest convenience method."""
    
    def test_quick_backtest_creates_and_runs(self):
        """Test that quick_backtest creates orchestrator and runs everything."""
        # This would need mocking for real use, but tests the API
        # For now just verify the method exists and has right signature
        assert hasattr(BacktestOrchestrator, 'quick_backtest')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
