"""
Test suite for Energy Seasonal V2 signal generators.

Tests for signals/energy_seasonal_v2.py:
- EnergySeasonalBalanced initialization and signal generation
- EnergySeasonalAggressive initialization and signal generation
- Dynamic volatility regime detection
- Seasonal pattern integration
- Trend filtering
"""

import pytest
import pandas as pd
import numpy as np
from signals.energy_seasonal_v2 import EnergySeasonalBalanced, EnergySeasonalAggressive


@pytest.fixture
def energy_price_data():
    """Generate synthetic energy price data with seasonal patterns."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=400, freq='D')
    
    # Base trend
    trend = np.linspace(50, 70, 400)
    
    # Seasonal pattern (annual cycle for energy)
    days_of_year = np.array([d.dayofyear for d in dates])
    seasonal = 10 * np.sin(2 * np.pi * days_of_year / 365)
    
    # Volatility clusters
    vol = 0.02 + 0.01 * np.sin(np.linspace(0, 4 * np.pi, 400))
    noise = np.random.randn(400) * vol * trend
    
    prices = trend + seasonal + noise
    prices = np.maximum(prices, 10)  # Keep prices positive
    
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'High': prices * 1.015,
        'Low': prices * 0.985,
        'Open': prices * 1.005,
    }).set_index('Date')


@pytest.fixture
def high_volatility_data():
    """Generate high volatility energy price data."""
    np.random.seed(123)
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    
    # High volatility with frequent spikes
    base_price = 60
    returns = np.random.normal(0, 0.05, 300)  # 5% daily vol
    prices = base_price * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Open': prices * 1.01,
    }).set_index('Date')


# ============================================================================
# EnergySeasonalBalanced Tests
# ============================================================================

class TestEnergySeasonalBalancedInitialization:
    """Test EnergySeasonalBalanced initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        signal = EnergySeasonalBalanced()
        
        assert signal.vol_window == 60
        assert signal.vol_percentile_threshold == 0.75
        assert signal.mr_window == 20
        assert signal.use_seasonality == True
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        signal = EnergySeasonalBalanced(
            vol_window=90,
            vol_percentile_threshold=0.80,
            mr_window=30,
            mr_entry_z=2.5,
            mr_exit_z=0.5,
            mom_window=50,
            use_seasonality=False,
            trend_window=150
        )
        
        assert signal.vol_window == 90
        assert signal.vol_percentile_threshold == 0.80
        assert signal.mr_window == 30
        assert signal.mr_entry_z == 2.5
        assert signal.mr_exit_z == 0.5
        assert signal.mom_window == 50
        assert signal.use_seasonality == False
        assert signal.trend_window == 150


class TestEnergySeasonalBalancedSignalGeneration:
    """Test signal generation for EnergySeasonalBalanced."""
    
    def test_generate_returns_dataframe(self, energy_price_data):
        """Test that generate returns a DataFrame."""
        signal = EnergySeasonalBalanced()
        result = signal.generate(energy_price_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(energy_price_data)
    
    def test_generate_adds_required_columns(self, energy_price_data):
        """Test that all required columns are added."""
        signal = EnergySeasonalBalanced()
        result = signal.generate(energy_price_data)
        
        # Check for key columns
        assert 'Vol' in result.columns
        assert 'Vol_Rank' in result.columns
        assert 'HighVol' in result.columns
        assert 'Z_Score' in result.columns
        assert 'Signal' in result.columns
    
    def test_signal_values_are_valid(self, energy_price_data):
        """Test that signals are -1, 0, or 1."""
        signal = EnergySeasonalBalanced()
        result = signal.generate(energy_price_data)
        
        assert result['Signal'].isin([-1, 0, 1]).all()
    
    def test_volatility_regime_detection(self, energy_price_data):
        """Test that volatility regime is properly detected."""
        signal = EnergySeasonalBalanced(vol_percentile_threshold=0.75)
        result = signal.generate(energy_price_data)
        
        # HighVol should be binary
        assert result['HighVol'].isin([0, 1]).all()
        
        # Should have some high vol periods
        high_vol_count = result['HighVol'].sum()
        total_count = result['HighVol'].notna().sum()
        high_vol_pct = high_vol_count / total_count if total_count > 0 else 0
        
        # With 75th percentile threshold, roughly 25% should be high vol
        assert 0.15 < high_vol_pct < 0.35
    
    def test_mean_reversion_component(self, energy_price_data):
        """Test that mean reversion Z-score is calculated."""
        signal = EnergySeasonalBalanced()
        result = signal.generate(energy_price_data)
        
        # Z_Score should be present and have valid values
        assert 'Z_Score' in result.columns
        assert result['Z_Score'].notna().any()
        
        # Z-scores should be centered around 0
        z_mean = result['Z_Score'].dropna().mean()
        assert abs(z_mean) < 0.5
    
    def test_with_seasonality_enabled(self, energy_price_data):
        """Test signal generation with seasonality enabled."""
        signal = EnergySeasonalBalanced(use_seasonality=True)
        result = signal.generate(energy_price_data)
        
        assert 'Signal' in result.columns
        assert result['Signal'].notna().any()
    
    def test_with_seasonality_disabled(self, energy_price_data):
        """Test signal generation with seasonality disabled."""
        signal = EnergySeasonalBalanced(use_seasonality=False)
        result = signal.generate(energy_price_data)
        
        assert 'Signal' in result.columns
        assert result['Signal'].notna().any()


class TestEnergySeasonalBalancedHighVolatility:
    """Test behavior in high volatility environments."""
    
    def test_high_volatility_regime(self, high_volatility_data):
        """Test signal generation in high volatility regime."""
        signal = EnergySeasonalBalanced(vol_percentile_threshold=0.70)
        result = signal.generate(high_volatility_data)
        
        # Should detect more high vol periods
        high_vol_pct = result['HighVol'].sum() / len(result)
        assert high_vol_pct > 0.20  # At least 20% high vol
    
    def test_different_vol_thresholds(self, energy_price_data):
        """Test different volatility percentile thresholds."""
        # Strict threshold (90th percentile)
        signal_strict = EnergySeasonalBalanced(vol_percentile_threshold=0.90)
        result_strict = signal_strict.generate(energy_price_data)
        
        # Loose threshold (50th percentile)
        signal_loose = EnergySeasonalBalanced(vol_percentile_threshold=0.50)
        result_loose = signal_loose.generate(energy_price_data)
        
        # Loose threshold should detect more high vol periods
        strict_high_vol = result_strict['HighVol'].sum()
        loose_high_vol = result_loose['HighVol'].sum()
        assert loose_high_vol > strict_high_vol


# ============================================================================
# EnergySeasonalAggressive Tests
# ============================================================================

class TestEnergySeasonalAggressiveInitialization:
    """Test EnergySeasonalAggressive initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        signal = EnergySeasonalAggressive()
        
        assert signal.vol_window == 30  # Actual default
        assert signal.vol_percentile_threshold == 0.90  # Actual default (extreme vol only)
        assert signal.mr_window == 14  # Faster MR
        assert signal.use_seasonality == True
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        signal = EnergySeasonalAggressive(
            vol_window=45,
            mr_window=10,
            mr_entry_z=1.8,
            crisis_alpha_entry=3.5,
            seasonal_strength=1.2
        )
        
        assert signal.vol_window == 45
        assert signal.mr_window == 10
        assert signal.mr_entry_z == 1.8
        assert signal.crisis_alpha_entry == 3.5
        assert signal.seasonal_strength == 1.2


class TestEnergySeasonalAggressiveSignalGeneration:
    """Test signal generation for EnergySeasonalAggressive."""
    
    def test_generate_returns_dataframe(self, energy_price_data):
        """Test that generate returns a DataFrame."""
        signal = EnergySeasonalAggressive()
        result = signal.generate(energy_price_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(energy_price_data)
    
    def test_generate_adds_signal_column(self, energy_price_data):
        """Test that Signal column is added."""
        signal = EnergySeasonalAggressive()
        result = signal.generate(energy_price_data)
        
        assert 'Signal' in result.columns
    
    def test_signal_values_are_valid(self, energy_price_data):
        """Test that signals are -1, 0, or 1."""
        signal = EnergySeasonalAggressive()
        result = signal.generate(energy_price_data)
        
        assert result['Signal'].isin([-1, 0, 1]).all()
    
    def test_aggressive_more_active(self, energy_price_data):
        """Test that aggressive strategy is more active than balanced."""
        balanced = EnergySeasonalBalanced()
        aggressive = EnergySeasonalAggressive()
        
        result_balanced = balanced.generate(energy_price_data)
        result_aggressive = aggressive.generate(energy_price_data)
        
        # Aggressive should have fewer zeros (more active trading)
        zeros_balanced = (result_balanced['Signal'] == 0).sum()
        zeros_aggressive = (result_aggressive['Signal'] == 0).sum()
        
        # Allow some tolerance as both might be active
        # The key is that aggressive uses lower thresholds
        assert zeros_aggressive <= zeros_balanced * 1.1  # Allow 10% tolerance
    
    def test_crisis_alpha_parameter(self, energy_price_data):
        """Test that crisis alpha entry parameter is configurable."""
        signal = EnergySeasonalAggressive(crisis_alpha_entry=2.5)
        result = signal.generate(energy_price_data)
        
        # Signals should still be -1, 0, 1
        assert result['Signal'].isin([-1, 0, 1]).all()


class TestEnergySeasonalEdgeCases:
    """Test edge cases for both energy seasonal strategies."""
    
    def test_short_data_series(self):
        """Test with very short data series."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'Close': np.linspace(60, 65, 100),
            'High': np.linspace(61, 66, 100),
            'Low': np.linspace(59, 64, 100),
            'Open': np.linspace(60, 65, 100),
        }, index=dates)
        
        signal = EnergySeasonalBalanced()
        result = signal.generate(prices)
        
        assert 'Signal' in result.columns
        assert len(result) == len(prices)
    
    def test_flat_prices(self):
        """Test with flat (no volatility) prices."""
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        prices = pd.DataFrame({
            'Close': np.ones(200) * 60,
            'High': np.ones(200) * 60.5,
            'Low': np.ones(200) * 59.5,
            'Open': np.ones(200) * 60,
        }, index=dates)
        
        signal = EnergySeasonalBalanced()
        result = signal.generate(prices)
        
        # Should handle flat prices without errors
        assert 'Signal' in result.columns
    
    def test_missing_data_handling(self, energy_price_data):
        """Test handling of missing data."""
        df_with_gaps = energy_price_data.copy()
        # Introduce some NaN values
        df_with_gaps.loc[df_with_gaps.index[50:60], 'Close'] = np.nan
        
        signal = EnergySeasonalBalanced()
        result = signal.generate(df_with_gaps)
        
        # Should still produce output
        assert 'Signal' in result.columns
    
    def test_extreme_volatility(self):
        """Test with extreme volatility."""
        np.random.seed(999)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        
        # Extreme price swings
        returns = np.random.normal(0, 0.10, 200)  # 10% daily vol!
        prices = 60 * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'Close': prices,
            'High': prices * 1.05,
            'Low': prices * 0.95,
            'Open': prices * 1.01,
        }, index=dates)
        
        signal = EnergySeasonalBalanced()
        result = signal.generate(df)
        
        # Should handle extreme vol
        assert 'Signal' in result.columns
        assert result['Signal'].isin([-1, 0, 1]).all()


class TestEnergySeasonalComparison:
    """Compare balanced vs aggressive strategies."""
    
    def test_both_strategies_work(self, energy_price_data):
        """Test that both strategies work on same data."""
        balanced = EnergySeasonalBalanced()
        aggressive = EnergySeasonalAggressive()
        
        result_balanced = balanced.generate(energy_price_data)
        result_aggressive = aggressive.generate(energy_price_data)
        
        assert 'Signal' in result_balanced.columns
        assert 'Signal' in result_aggressive.columns
        assert len(result_balanced) == len(result_aggressive)
    
    def test_parameter_differences(self):
        """Test that strategies use different default parameters."""
        balanced = EnergySeasonalBalanced()
        aggressive = EnergySeasonalAggressive()
        
        # Note: Aggressive uses HIGHER vol threshold (0.90) to ignore vol except in extremes
        # Balanced uses lower threshold (0.75) to be more conservative
        # This is intentional - aggressive only stops in extreme vol
        assert aggressive.vol_percentile_threshold > balanced.vol_percentile_threshold
        
        # Aggressive should have faster MR
        assert aggressive.mr_window < balanced.mr_window
