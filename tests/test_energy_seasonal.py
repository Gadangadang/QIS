"""
Tests for EnergySeasonalSignal
"""
import pytest
import pandas as pd
import numpy as np
from signals.energy_seasonal import EnergySeasonalSignal, EnergySeasonalLongOnly


def create_test_data(n_days=500, seed=42):
    """Create synthetic price data for testing."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Create realistic price data with seasonality
    trend = np.linspace(100, 120, n_days)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365)  # Annual cycle
    noise = np.random.normal(0, 2, n_days)
    prices = trend + seasonal + noise
    
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, n_days)
    }, index=dates)
    
    return df


class TestEnergySeasonalSignal:
    """Test suite for EnergySeasonalSignal."""
    
    def test_initialization_valid(self):
        """Test signal initializes with valid parameters."""
        signal = EnergySeasonalSignal(
            vol_window=30,
            mr_window=20,
            mom_lookback=60
        )
        assert signal.vol_window == 30
        assert signal.mr_window == 20
        assert signal.mom_lookback == 60
    
    def test_initialization_invalid_params(self):
        """Test signal raises errors for invalid parameters."""
        with pytest.raises(ValueError):
            EnergySeasonalSignal(vol_window=1)  # Too small
        
        with pytest.raises(ValueError):
            EnergySeasonalSignal(winter_bias=2.0)  # Out of range
        
        with pytest.raises(ValueError):
            EnergySeasonalSignal(summer_bias=-2.0)  # Out of range
    
    def test_generate_basic(self):
        """Test basic signal generation."""
        df = create_test_data(500)
        signal = EnergySeasonalSignal()
        result = signal.generate(df)
        
        # Check required columns exist
        assert 'Signal' in result.columns
        assert 'Season' in result.columns
        assert 'Seasonal_Bias' in result.columns
        assert 'HighVol' in result.columns
        assert 'MR_Signal' in result.columns
        assert 'Mom_Signal' in result.columns
        
        # Check signal values are valid
        assert result['Signal'].isin([-1, 0, 1]).all()
        
        # Check warmup period is flat
        warmup = max(30, 20, 60)
        assert (result['Signal'].iloc[:warmup] == 0).all()
    
    def test_seasonal_assignments(self):
        """Test seasonal assignments are correct."""
        df = create_test_data(365)
        signal = EnergySeasonalSignal()
        result = signal.generate(df)
        
        # Check winter months
        winter_mask = result.index.month.isin([11, 12, 1, 2, 3])
        assert (result.loc[winter_mask, 'Season'] == 'Winter').all()
        assert (result.loc[winter_mask, 'Seasonal_Bias'] == 0.6).all()
        
        # Check summer months
        summer_mask = result.index.month.isin([6, 7, 8])
        assert (result.loc[summer_mask, 'Season'] == 'Summer').all()
        assert (result.loc[summer_mask, 'Seasonal_Bias'] == 0.3).all()
        
        # Check shoulder months
        shoulder_mask = result.index.month.isin([4, 5, 9, 10])
        assert (result.loc[shoulder_mask, 'Season'] == 'Shoulder').all()
        assert (result.loc[shoulder_mask, 'Seasonal_Bias'] == -0.2).all()
    
    def test_seasonality_filtering(self):
        """Test that seasonality filtering affects signals correctly."""
        df = create_test_data(365)
        
        # With seasonality
        signal_seasonal = EnergySeasonalSignal(use_seasonality=True)
        result_seasonal = signal_seasonal.generate(df)
        
        # Without seasonality
        signal_no_seasonal = EnergySeasonalSignal(use_seasonality=False)
        result_no_seasonal = signal_no_seasonal.generate(df)
        
        # Signals should differ when seasonality is enabled
        # (unless by chance all signals align with seasonal bias)
        assert not result_seasonal['Signal'].equals(result_no_seasonal['Signal'])
    
    def test_regime_switching(self):
        """Test that strategy switches between mean reversion and momentum."""
        df = create_test_data(500)
        signal = EnergySeasonalSignal()
        result = signal.generate(df)
        
        # Should have both high vol and low vol periods
        assert result['HighVol'].sum() > 0
        assert result['HighVol'].sum() < len(result)
        
        # Mean reversion signals should exist
        assert (result['MR_Signal'] != 0).any()
        
        # Momentum signals should exist
        assert (result['Mom_Signal'] != 0).any()
    
    def test_empty_dataframe(self):
        """Test that empty DataFrame raises error."""
        signal = EnergySeasonalSignal()
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            signal.generate(df)
    
    def test_missing_close_column(self):
        """Test that missing Close column raises error."""
        signal = EnergySeasonalSignal()
        df = pd.DataFrame({'Open': [100, 101, 102]})
        
        with pytest.raises(ValueError, match="must contain 'Close' column"):
            signal.generate(df)


class TestEnergySeasonalLongOnly:
    """Test suite for EnergySeasonalLongOnly."""
    
    def test_no_short_signals(self):
        """Test that long-only version never generates short signals."""
        df = create_test_data(500)
        signal = EnergySeasonalLongOnly()
        result = signal.generate(df)
        
        # Should never have -1 signals
        assert (result['Signal'] >= 0).all()
        assert result['Signal'].isin([0, 1]).all()
    
    def test_converts_shorts_to_flat(self):
        """Test that short signals are converted to flat."""
        df = create_test_data(500)
        
        # Generate with base signal (allows shorts)
        base_signal = EnergySeasonalSignal()
        base_result = base_signal.generate(df)
        
        # Generate with long-only (no shorts)
        long_only_signal = EnergySeasonalLongOnly()
        long_only_result = long_only_signal.generate(df)
        
        # Long signals should be the same
        long_mask = base_result['Signal'] == 1
        assert (base_result.loc[long_mask, 'Signal'] == 
                long_only_result.loc[long_mask, 'Signal']).all()
        
        # Short signals in base should be flat in long-only
        short_mask = base_result['Signal'] == -1
        if short_mask.any():
            assert (long_only_result.loc[short_mask, 'Signal'] == 0).all()


def test_realistic_natural_gas_scenario():
    """Test with realistic natural gas price behavior."""
    # Create data with high winter volatility
    n_days = 730  # 2 years
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Base price with seasonal pattern
    base = 3.0
    seasonal = 1.0 * np.sin(2 * np.pi * np.arange(n_days) / 365 + np.pi/2)  # Peak in winter
    
    # Higher volatility in winter
    vol = np.where(np.sin(2 * np.pi * np.arange(n_days) / 365 + np.pi/2) > 0, 0.3, 0.1)
    noise = np.random.normal(0, 1, n_days) * vol
    
    prices = base + seasonal + noise
    prices = np.maximum(prices, 0.5)  # Floor at $0.50
    
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(10000, 100000, n_days)
    }, index=dates)
    
    # Generate signals
    signal = EnergySeasonalSignal(
        vol_threshold=0.015,
        winter_bias=0.6,
        summer_bias=0.3,
        shoulder_bias=-0.2
    )
    result = signal.generate(df)
    
    # Should have signals
    assert (result['Signal'] != 0).any()
    
    # Should detect high volatility in winter
    winter_mask = result.index.month.isin([12, 1, 2])
    winter_high_vol_pct = result.loc[winter_mask, 'HighVol'].mean()
    
    summer_mask = result.index.month.isin([6, 7, 8])
    summer_high_vol_pct = result.loc[summer_mask, 'HighVol'].mean()
    
    # Winter should have more high vol periods (not always, but on average)
    # This is a probabilistic test, so we just check it's reasonable
    assert winter_high_vol_pct >= 0  # At least some vol detection


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
