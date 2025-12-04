"""
Tests for enhanced BacktestOrchestrator features:
- Config support
- capital_pct allocation
- Position sizer selection
- OOS split and backtesting
- Walk-forward integration
- HTML export

Run with: pytest tests/test_orchestrator_enhanced.py -v
"""

import pytest
import pandas as pd
import numpy as np
import os

from core.backtest_orchestrator import BacktestOrchestrator
from signals.momentum import MomentumSignalV2


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        'assets': ['ES', 'NQ'],
        'total_capital': 500_000,
        'oos_split': 0.20,
        'date_range': ('2020-01-01', '2024-12-31'),
        'use_futures_sizing': False
    }


@pytest.fixture
def sample_data():
    """Create sample price data."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    data = {}
    for ticker in ['ES', 'NQ']:
        base_price = 4500 if ticker == 'ES' else 15000
        data[ticker] = pd.DataFrame({
            'Open': base_price + np.cumsum(np.random.randn(500)) * 10,
            'High': base_price + 20 + np.cumsum(np.random.randn(500)) * 10,
            'Low': base_price - 20 + np.cumsum(np.random.randn(500)) * 10,
            'Close': base_price + np.cumsum(np.random.randn(500)) * 10,
            'Volume': 1000000 + np.random.randint(-100000, 100000, 500)
        }, index=dates)
    
    return data


class TestConfigSupport:
    """Test configuration dictionary support."""
    
    def test_init_with_config(self, sample_config):
        """Test initialization with config dict."""
        orch = BacktestOrchestrator(config=sample_config)
        
        assert orch.total_capital == 500_000
        assert orch.oos_split == 0.20
        assert orch.use_futures_sizing == False
    
    def test_init_without_config(self):
        """Test initialization without config."""
        orch = BacktestOrchestrator()
        
        assert orch.total_capital == 0.0
        assert orch.oos_split == 0.0
        assert orch.allocated_capital == 0.0


class TestCapitalPctAllocation:
    """Test percentage-based capital allocation."""
    
    def test_add_strategy_with_capital_pct(self, sample_config, sample_data):
        """Test adding strategy with capital_pct."""
        orch = BacktestOrchestrator(config=sample_config)
        orch.prices = sample_data
        orch._data_loaded = True
        
        orch.add_strategy(
            name='Strategy1',
            signal_generator=MomentumSignalV2(lookback=20),
            assets=['ES'],
            capital_pct=0.40  # 40% of 500k = 200k
        )
        
        assert len(orch.strategies) == 1
        assert orch.strategies[0].capital == 200_000
        assert orch.allocated_capital == 0.40
    
    def test_capital_pct_validation(self, sample_config, sample_data):
        """Test capital_pct validation."""
        orch = BacktestOrchestrator(config=sample_config)
        orch.prices = sample_data
        orch._data_loaded = True
        
        # Add 60% allocation
        orch.add_strategy('S1', MomentumSignalV2(), ['ES'], capital_pct=0.60)
        
        # Try to add 50% more (would exceed 100%)
        with pytest.raises(ValueError, match="Total capital allocation would exceed 100%"):
            orch.add_strategy('S2', MomentumSignalV2(), ['NQ'], capital_pct=0.50)
    
    def test_cannot_specify_both_capital_and_pct(self, sample_config, sample_data):
        """Test error when specifying both capital and capital_pct."""
        orch = BacktestOrchestrator(config=sample_config)
        orch.prices = sample_data
        orch._data_loaded = True
        
        with pytest.raises(ValueError, match="Cannot specify both 'capital' and 'capital_pct'"):
            orch.add_strategy(
                'S1',
                MomentumSignalV2(),
                ['ES'],
                capital=100000,
                capital_pct=0.20
            )
    
    def test_capital_pct_requires_total_capital(self, sample_data):
        """Test that capital_pct requires total_capital in config."""
        orch = BacktestOrchestrator()  # No config
        orch.prices = sample_data
        orch._data_loaded = True
        
        with pytest.raises(ValueError, match="Cannot use capital_pct without setting total_capital"):
            orch.add_strategy('S1', MomentumSignalV2(), ['ES'], capital_pct=0.40)


class TestPositionSizerSelection:
    """Test position sizer type selection."""
    
    def test_fixed_sizer_creation(self, sample_data):
        """Test creating fixed fractional sizer."""
        orch = BacktestOrchestrator()
        orch.prices = sample_data
        orch._data_loaded = True
        
        orch.add_strategy(
            'S1',
            MomentumSignalV2(),
            ['ES'],
            capital=100000,
            position_sizer_type='fixed'
        )
        
        from core.portfolio.position_sizers import FixedFractionalSizer
        sizer = orch._create_position_sizer(orch.strategies[0])
        assert isinstance(sizer, FixedFractionalSizer)
    
    def test_atr_sizer_creation(self, sample_data):
        """Test creating ATR sizer."""
        orch = BacktestOrchestrator()
        orch.prices = sample_data
        orch._data_loaded = True
        
        orch.add_strategy(
            'S1',
            MomentumSignalV2(),
            ['ES'],
            capital=100000,
            position_sizer_type='atr',
            position_sizer_params={'atr_multiplier': 2.0}
        )
        
        from core.portfolio.position_sizers import ATRSizer
        sizer = orch._create_position_sizer(orch.strategies[0])
        assert isinstance(sizer, ATRSizer)
    
    def test_volatility_sizer_creation(self, sample_data):
        """Test creating volatility sizer."""
        orch = BacktestOrchestrator()
        orch.prices = sample_data
        orch._data_loaded = True
        
        orch.add_strategy(
            'S1',
            MomentumSignalV2(),
            ['ES'],
            capital=100000,
            position_sizer_type='volatility'
        )
        
        from core.portfolio.position_sizers import VolatilityScaledSizer
        sizer = orch._create_position_sizer(orch.strategies[0])
        assert isinstance(sizer, VolatilityScaledSizer)
    
    def test_invalid_sizer_type_raises_error(self, sample_data):
        """Test error on invalid sizer type."""
        orch = BacktestOrchestrator()
        orch.prices = sample_data
        orch._data_loaded = True
        
        orch.add_strategy(
            'S1',
            MomentumSignalV2(),
            ['ES'],
            capital=100000,
            position_sizer_type='invalid_type'
        )
        
        with pytest.raises(ValueError, match="Unknown position_sizer_type"):
            orch._create_position_sizer(orch.strategies[0])


class TestOOSSplit:
    """Test out-of-sample data splitting."""
    
    def test_split_train_test_data(self, sample_config, sample_data):
        """Test train/test splitting."""
        orch = BacktestOrchestrator(config=sample_config)
        orch.prices = sample_data
        orch._data_loaded = True
        
        orch.split_train_test_data(verbose=False)
        
        # Check split happened
        assert len(orch.prices_train) == 2  # ES and NQ
        assert len(orch.prices_test) == 2
        
        # Check sizes (80% train, 20% test)
        for ticker in ['ES', 'NQ']:
            total_len = len(sample_data[ticker])
            train_len = len(orch.prices_train[ticker])
            test_len = len(orch.prices_test[ticker])
            
            assert train_len == int(total_len * 0.80)
            assert test_len == total_len - train_len
    
    def test_split_with_custom_oos_pct(self, sample_data):
        """Test splitting with custom OOS percentage."""
        orch = BacktestOrchestrator()
        orch.prices = sample_data
        orch._data_loaded = True
        
        orch.split_train_test_data(oos_pct=0.30, verbose=False)
        
        # Check 70/30 split
        for ticker in ['ES', 'NQ']:
            total_len = len(sample_data[ticker])
            train_len = len(orch.prices_train[ticker])
            
            assert train_len == int(total_len * 0.70)
    
    def test_split_requires_positive_oos_pct(self, sample_data):
        """Test that oos_pct must be positive."""
        orch = BacktestOrchestrator()
        orch.prices = sample_data
        orch._data_loaded = True
        
        with pytest.raises(ValueError, match="oos_pct must be > 0"):
            orch.split_train_test_data(oos_pct=0.0)


class TestOOSBacktest:
    """Test out-of-sample backtesting."""
    
    def test_run_oos_backtest(self, sample_config, sample_data):
        """Test running OOS backtest."""
        orch = BacktestOrchestrator(config=sample_config)
        orch.prices = sample_data
        orch._data_loaded = True
        
        # Split data
        orch.split_train_test_data(verbose=False)
        
        # Add strategy and run train backtest
        orch.add_strategy('S1', MomentumSignalV2(), ['ES'], capital=100000)
        orch.generate_signals(verbose=False)
        orch.run_backtests(verbose=False)
        
        # Run OOS backtest
        oos_results = orch.run_oos_backtest(verbose=False)
        
        assert 'S1' in oos_results
        assert 'result' in oos_results['S1']
        assert orch._oos_run == True
    
    def test_oos_requires_split(self, sample_data):
        """Test that OOS requires split_train_test_data."""
        orch = BacktestOrchestrator()
        orch.prices = sample_data
        orch._data_loaded = True
        
        orch.add_strategy('S1', MomentumSignalV2(), ['ES'], capital=100000)
        orch.generate_signals(verbose=False)
        
        with pytest.raises(RuntimeError, match="Must call split_train_test_data"):
            orch.run_oos_backtest(verbose=False)


class TestWalkForwardIntegration:
    """Test walk-forward optimization integration."""
    
    def test_run_walkforward(self, sample_data):
        """Test walk-forward optimization."""
        orch = BacktestOrchestrator()
        orch.prices = sample_data
        orch._data_loaded = True
        
        optimizer = orch.run_walkforward(
            signal_class=MomentumSignalV2,
            param_grid={'lookback': [20, 30]},
            assets=['ES'],
            train_pct=0.60,
            test_pct=0.20,
            initial_capital=100000,
            verbose=False
        )
        
        assert optimizer is not None
        assert len(optimizer.periods) > 0
    
    def test_walkforward_single_asset_only(self, sample_data):
        """Test that walk-forward currently requires single asset."""
        orch = BacktestOrchestrator()
        orch.prices = sample_data
        orch._data_loaded = True
        
        with pytest.raises(ValueError, match="single asset only"):
            orch.run_walkforward(
                signal_class=MomentumSignalV2,
                param_grid={'lookback': [20]},
                assets=['ES', 'NQ'],  # Multiple assets
                verbose=False
            )


class TestHTMLExport:
    """Test HTML dashboard export."""
    
    def test_export_html_dashboard(self, sample_data, tmp_path):
        """Test HTML export functionality."""
        orch = BacktestOrchestrator()
        orch.prices = sample_data
        orch._data_loaded = True
        
        orch.add_strategy('S1', MomentumSignalV2(), ['ES'], capital=100000)
        orch.generate_signals(verbose=False)
        orch.run_backtests(verbose=False)
        
        # Export to temp directory
        output_dir = str(tmp_path / "html")
        filepath = orch.export_html_dashboard(
            output_dir=output_dir,
            filename_prefix='test'
        )
        
        # Check file was created
        assert os.path.exists(filepath)
        assert filepath.endswith('.html')
        
        # Check content
        with open(filepath, 'r') as f:
            content = f.read()
            assert len(content) > 0
    
    def test_export_with_oos_results(self, sample_config, sample_data, tmp_path):
        """Test HTML export with OOS results."""
        orch = BacktestOrchestrator(config=sample_config)
        orch.prices = sample_data
        orch._data_loaded = True
        
        # Run full workflow with OOS
        orch.split_train_test_data(verbose=False)
        orch.add_strategy('S1', MomentumSignalV2(), ['ES'], capital=100000)
        orch.generate_signals(verbose=False)
        orch.run_backtests(verbose=False)
        orch.run_oos_backtest(verbose=False)
        
        # Export with OOS
        output_dir = str(tmp_path / "html")
        filepath = orch.export_html_dashboard(
            output_dir=output_dir,
            include_oos=True
        )
        
        # Check file exists and has OOS content
        with open(filepath, 'r') as f:
            content = f.read()
            assert 'Out-of-Sample Results' in content
    
    def test_export_requires_backtests(self):
        """Test that export requires backtests to be run."""
        orch = BacktestOrchestrator()
        
        with pytest.raises(RuntimeError, match="Must run backtests first"):
            orch.export_html_dashboard()
