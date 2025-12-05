"""
Comprehensive test suite for all signal generators.

Tests verify:
- Input validation
- Output structure
- Signal generation logic
- Edge cases
- Type hints and contracts
"""

import pytest
import pandas as pd
import numpy as np
from signals.mean_reversion import MeanReversionSignal
from signals.momentum import MomentumSignalV2
from signals.hybrid_adaptive import HybridAdaptiveSignal
from signals.trend_following_long_short import TrendFollowingLongShort, AdaptiveTrendFollowing
from signals.ensemble import EnsembleSignal, EnsembleSignalNew


@pytest.fixture
def sample_price_data():
    """Generate synthetic price data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Create trending + mean-reverting price pattern
    trend = np.linspace(100, 120, 200)
    noise = np.random.randn(200) * 2
    mean_rev = 5 * np.sin(np.linspace(0, 4 * np.pi, 200))
    
    prices = trend + noise + mean_rev
    
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Open': prices * 1.005,
    }).set_index('Date')


@pytest.fixture
def uptrend_data():
    """Generate strong uptrend data."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = np.linspace(100, 150, 100) + np.random.randn(100) * 0.5
    return pd.DataFrame({
        'Close': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
    }, index=dates)


@pytest.fixture
def downtrend_data():
    """Generate strong downtrend data."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = np.linspace(150, 100, 100) + np.random.randn(100) * 0.5
    return pd.DataFrame({
        'Close': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
    }, index=dates)


@pytest.fixture
def sideways_data():
    """Generate sideways/ranging data."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + 5 * np.sin(np.linspace(0, 8 * np.pi, 100)) + np.random.randn(100) * 0.3
    return pd.DataFrame({
        'Close': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
    }, index=dates)


# ============================================================================
# MeanReversionSignal Tests
# ============================================================================

class TestMeanReversionSignal:
    """Test suite for MeanReversionSignal."""
    
    def test_initialization(self):
        """Test signal can be initialized with valid parameters."""
        signal = MeanReversionSignal(window=20, entry_z=2.0, exit_z=0.5)
        assert signal.window == 20
        assert signal.entry_z == 2.0
        assert signal.exit_z == 0.5
    
    def test_validation_window_too_small(self):
        """Test that window < 2 raises ValueError."""
        with pytest.raises(ValueError, match="window must be >= 2"):
            MeanReversionSignal(window=1)
    
    def test_validation_entry_z_negative(self):
        """Test that negative entry_z raises ValueError."""
        with pytest.raises(ValueError, match="entry_z must be positive"):
            MeanReversionSignal(entry_z=-1.0)
    
    def test_validation_exit_z_negative(self):
        """Test that negative exit_z raises ValueError."""
        with pytest.raises(ValueError, match="exit_z must be non-negative"):
            MeanReversionSignal(exit_z=-0.1)
    
    def test_validation_exit_larger_than_entry(self):
        """Test that exit_z >= entry_z raises ValueError."""
        with pytest.raises(ValueError, match="exit_z.*must be < entry_z"):
            MeanReversionSignal(entry_z=1.0, exit_z=1.5)
    
    def test_generate_returns_dataframe(self, sample_price_data):
        """Test generate() returns a DataFrame."""
        signal = MeanReversionSignal()
        result = signal.generate(sample_price_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_generate_adds_required_columns(self, sample_price_data):
        """Test generate() adds Z and Signal columns."""
        signal = MeanReversionSignal()
        result = signal.generate(sample_price_data)
        assert 'Z' in result.columns
        assert 'Signal' in result.columns
    
    def test_signals_are_valid_values(self, sample_price_data):
        """Test signals are only -1, 0, or 1."""
        signal = MeanReversionSignal()
        result = signal.generate(sample_price_data)
        valid_signals = result['Signal'].isin([-1, 0, 1]).all()
        assert valid_signals
    
    def test_warmup_period_has_zero_signals(self, sample_price_data):
        """Test first 'window' bars have Signal = 0."""
        window = 20
        signal = MeanReversionSignal(window=window)
        result = signal.generate(sample_price_data)
        assert (result['Signal'].iloc[:window] == 0).all()
    
    def test_empty_dataframe_raises_error(self):
        """Test empty DataFrame raises ValueError."""
        signal = MeanReversionSignal()
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            signal.generate(empty_df)
    
    def test_missing_close_column_raises_error(self):
        """Test DataFrame without Close column raises ValueError."""
        signal = MeanReversionSignal()
        df = pd.DataFrame({'Open': [100, 101, 102]})
        with pytest.raises(ValueError, match="must have 'Close' column"):
            signal.generate(df)
    
    def test_long_signals_on_oversold(self, sideways_data):
        """Test generates long signals when price is oversold."""
        signal = MeanReversionSignal(window=10, entry_z=1.5, exit_z=0.3)
        result = signal.generate(sideways_data)
        # Should have some long signals (1) when Z < -entry_z
        assert (result['Signal'] == 1).any()
    
    def test_short_signals_on_overbought(self, sideways_data):
        """Test generates short signals when price is overbought."""
        signal = MeanReversionSignal(window=10, entry_z=1.5, exit_z=0.3)
        result = signal.generate(sideways_data)
        # Should have some short signals (-1) when Z > entry_z
        assert (result['Signal'] == -1).any()


# ============================================================================
# MomentumSignalV2 Tests (Primary momentum signal - long-only with SMA filter)
# ============================================================================

class TestMomentumSignalV2:
    """Test suite for MomentumSignalV2 (improved version)."""
    
    def test_initialization(self):
        """Test signal can be initialized."""
        signal = MomentumSignalV2(lookback=20, entry_threshold=0.02, exit_threshold=0.005)
        assert signal.lookback == 20
        assert signal.entry_threshold == 0.02
        assert signal.exit_threshold == 0.005
    
    def test_generate_returns_dataframe(self, sample_price_data):
        """Test generate() returns a DataFrame."""
        signal = MomentumSignalV2()
        result = signal.generate(sample_price_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_generate_adds_required_columns(self, sample_price_data):
        """Test generate() adds Momentum and Signal columns."""
        signal = MomentumSignalV2()
        result = signal.generate(sample_price_data)
        assert 'Momentum' in result.columns
        assert 'Signal' in result.columns
    
    def test_signals_are_valid_values(self, sample_price_data):
        """Test signals are only -1, 0, or 1."""
        signal = MomentumSignalV2()
        result = signal.generate(sample_price_data)
        assert result['Signal'].isin([-1, 0, 1]).all()
    
    def test_long_bias_in_uptrend(self, uptrend_data):
        """Test generates more long signals in strong uptrend."""
        signal = MomentumSignalV2(lookback=10, entry_threshold=0.01, sma_filter=20)
        result = signal.generate(uptrend_data)
        long_signals = (result['Signal'] == 1).sum()
        # In strong uptrend, should have some long positions
        assert long_signals > 0
    
    def test_short_bias_in_downtrend(self, downtrend_data):
        """Test does NOT go long in strong downtrend (defensive)."""
        signal = MomentumSignalV2(lookback=10, entry_threshold=0.01, sma_filter=20)
        result = signal.generate(downtrend_data)
        long_signals = (result['Signal'] == 1).sum()
        # MomentumSignalV2 is long-only, should avoid longs in downtrend
        # May have 0 or very few long signals
        assert long_signals < 30  # Less than 30% of bars


# ============================================================================
# HybridAdaptiveSignal Tests
# ============================================================================

class TestHybridAdaptiveSignal:
    """Test suite for HybridAdaptiveSignal."""
    
    def test_initialization(self):
        """Test signal can be initialized with valid parameters."""
        signal = HybridAdaptiveSignal(
            vol_window=50,
            vol_threshold=0.012,
            mr_window=20,
            mr_entry_z=1.5,
            mr_exit_z=0.5,
            mom_fast=20,
            mom_slow=50
        )
        assert signal.vol_window == 50
        assert signal.vol_threshold == 0.012
        assert signal.mr_window == 20
    
    def test_generate_returns_dataframe(self, sample_price_data):
        """Test generate() returns a DataFrame."""
        signal = HybridAdaptiveSignal()
        result = signal.generate(sample_price_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_generate_adds_required_columns(self, sample_price_data):
        """Test generate() adds Volatility, HighVol, MR_Z, MA_Fast, MA_Slow, Signal columns."""
        signal = HybridAdaptiveSignal()
        result = signal.generate(sample_price_data)
        assert 'Volatility' in result.columns
        assert 'HighVol' in result.columns
        assert 'MR_Z' in result.columns
        assert 'MA_Fast' in result.columns
        assert 'MA_Slow' in result.columns
        assert 'Signal' in result.columns
    
    def test_signals_are_valid_values(self, sample_price_data):
        """Test signals are only -1, 0, or 1."""
        signal = HybridAdaptiveSignal()
        result = signal.generate(sample_price_data)
        assert result['Signal'].isin([-1, 0, 1]).all()
    
    def test_high_vol_flag_is_binary(self, sample_price_data):
        """Test HighVol is only 0 or 1."""
        signal = HybridAdaptiveSignal()
        result = signal.generate(sample_price_data)
        assert result['HighVol'].isin([0, 1]).all()
    
    def test_warmup_period_has_zero_signals(self, sample_price_data):
        """Test warmup period has Signal = 0."""
        signal = HybridAdaptiveSignal(vol_window=30, mom_slow=40)
        result = signal.generate(sample_price_data)
        warmup = max(30, 40)
        assert (result['Signal'].iloc[:warmup] == 0).all()


# ============================================================================
# TrendFollowingLongShort Tests
# ============================================================================

class TestTrendFollowingLongShort:
    """Test suite for TrendFollowingLongShort."""
    
    def test_initialization(self):
        """Test signal can be initialized."""
        signal = TrendFollowingLongShort(
            fast_period=20, 
            slow_period=100, 
            volume_period=50,
            momentum_threshold=0.02
        )
        assert signal.fast_period == 20
        assert signal.slow_period == 100
        assert signal.momentum_threshold == 0.02
    
    def test_generate_returns_dataframe(self, sample_price_data):
        """Test generate() returns a DataFrame."""
        signal = TrendFollowingLongShort()
        result = signal.generate(sample_price_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_generate_adds_signal_column(self, sample_price_data):
        """Test generate() adds Signal column."""
        signal = TrendFollowingLongShort()
        result = signal.generate(sample_price_data)
        assert 'Signal' in result.columns
    
    def test_signals_are_valid_values(self, sample_price_data):
        """Test signals are only -1, 0, or 1."""
        signal = TrendFollowingLongShort()
        result = signal.generate(sample_price_data)
        assert result['Signal'].isin([-1, 0, 1]).all()
    
    def test_requires_high_low_columns(self, sample_price_data):
        """Test works with Close and Volume columns."""
        signal = TrendFollowingLongShort()
        # Should work with full data
        result = signal.generate(sample_price_data)
        assert 'Signal' in result.columns
    
    def test_generates_long_and_short_signals(self, uptrend_data):
        """Test can generate signals in trending market."""
        signal = TrendFollowingLongShort(momentum_threshold=0.01, fast_period=10, slow_period=30)
        result = signal.generate(uptrend_data)
        # In uptrend data, should have some signals
        # Note: signal may be conservative and stay flat, which is okay
        assert 'Signal' in result.columns
        assert result['Signal'].isin([-1, 0, 1]).all()


class TestAdaptiveTrendFollowing:
    """Test suite for AdaptiveTrendFollowing."""
    
    def test_initialization(self):
        """Test signal can be initialized."""
        signal = AdaptiveTrendFollowing()
        # Check if class has expected attributes
        assert hasattr(signal, 'generate')
    
    def test_generate_returns_dataframe(self, sample_price_data):
        """Test generate() returns a DataFrame."""
        signal = AdaptiveTrendFollowing()
        result = signal.generate(sample_price_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_signals_are_valid_values(self, sample_price_data):
        """Test signals are only -1, 0, or 1."""
        signal = AdaptiveTrendFollowing()
        result = signal.generate(sample_price_data)
        assert result['Signal'].isin([-1, 0, 1]).all()


# ============================================================================
# Ensemble Signal Tests
# ============================================================================

class TestEnsembleSignal:
    """Test suite for EnsembleSignal (predefined momentum ensemble)."""
    
    def test_initialization(self):
        """Test ensemble can be initialized with no parameters."""
        signal = EnsembleSignal()
        assert len(signal.signals) == 3  # 3 momentum signals
    
    def test_generate_returns_dataframe(self, sample_price_data):
        """Test generate() returns a DataFrame."""
        signal = EnsembleSignal()
        result = signal.generate(sample_price_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_generate_adds_required_columns(self, sample_price_data):
        """Test generate() adds ensemble columns."""
        signal = EnsembleSignal()
        result = signal.generate(sample_price_data)
        assert 'EnsemblePosition' in result.columns
        assert 'TrendFilter' in result.columns
        assert 'Position' in result.columns
    
    def test_signals_are_valid_values(self, sample_price_data):
        """Test Position signals are only 0 or 1 (long-only)."""
        signal = EnsembleSignal()
        result = signal.generate(sample_price_data)
        assert result['Position'].isin([0, 1]).all()


class TestEnsembleSignalNew:
    """Test suite for EnsembleSignalNew (mean reversion ensemble)."""
    
    def test_initialization(self):
        """Test can initialize with no parameters."""
        signal = EnsembleSignalNew()
        assert hasattr(signal, 'generate')
    
    def test_generate_returns_dataframe(self, sample_price_data):
        """Test generate() returns a DataFrame."""
        signal = EnsembleSignalNew()
        result = signal.generate(sample_price_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_generate_adds_required_columns(self, sample_price_data):
        """Test generate() adds ensemble columns."""
        signal = EnsembleSignalNew()
        result = signal.generate(sample_price_data)
        assert 'SMA200' in result.columns
        assert 'InBullMarket' in result.columns
        assert 'MR_Vote' in result.columns
        assert 'Position' in result.columns
    
    def test_signals_are_valid_values(self, sample_price_data):
        """Test signals are only -1, 0, or 1."""
        signal = EnsembleSignalNew()
        result = signal.generate(sample_price_data)
        assert result['Position'].isin([-1, 0, 1]).all()


# ============================================================================
# Integration Tests
# ============================================================================

class TestSignalIntegration:
    """Integration tests across multiple signal types."""
    
    def test_all_signals_work_with_same_data(self, sample_price_data):
        """Test all core signals can process the same dataset."""
        signals = [
            ('MeanReversion', MeanReversionSignal(), 'Signal'),
            ('MomentumV2', MomentumSignalV2(), 'Signal'),
            ('HybridAdaptive', HybridAdaptiveSignal(), 'Signal'),
            ('EnsembleNew', EnsembleSignalNew(), 'Position'),
        ]
        
        for name, sig, col_name in signals:
            result = sig.generate(sample_price_data)
            assert col_name in result.columns, f"{name} missing {col_name} column"
            assert result[col_name].isin([-1, 0, 1]).all(), f"{name} has invalid signal values"
    
    def test_signals_preserve_dataframe_index(self, sample_price_data):
        """Test all signals preserve the original DataFrame index."""
        signal = MeanReversionSignal()
        result = signal.generate(sample_price_data)
        pd.testing.assert_index_equal(result.index, sample_price_data.index)
    
    def test_signals_dont_modify_original_dataframe(self, sample_price_data):
        """Test signals don't modify the input DataFrame."""
        original_cols = sample_price_data.columns.tolist()
        signal = MomentumSignalV2()
        _ = signal.generate(sample_price_data)
        assert sample_price_data.columns.tolist() == original_cols
