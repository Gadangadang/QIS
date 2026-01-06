"""
Tests for TAA Signal Generators.

Verifies momentum, carry, and ensemble signals for monthly TAA rebalancing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from signals.taa import (
    TimeSeriesMomentum, 
    CrossSectionalMomentum,
    RiskAdjustedMomentum,
    YieldCarry,
    TAAEnsembleSignal
)
from signals.taa.carry import RollYield, CombinedCarry
from signals.taa.ensemble import MultiAssetEnsemble


class TestTimeSeriesMomentum:
    """Test suite for TimeSeriesMomentum signal."""
    
    @pytest.fixture
    def monthly_prices(self):
        """Generate monthly price data for testing."""
        dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='MS')
        
        # Uptrend with noise
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.005, 0.02, len(dates))))
        
        return pd.DataFrame({
            'Close': prices
        }, index=dates)
    
    def test_initialization(self):
        """Test signal can be initialized with valid parameters."""
        signal = TimeSeriesMomentum(lookback_months=12, vol_adjust=True)
        assert signal.lookback_months == 12
        assert signal.vol_adjust is True
    
    def test_invalid_lookback_raises_error(self):
        """Test negative lookback raises ValueError."""
        with pytest.raises(ValueError, match="lookback_months must be positive"):
            TimeSeriesMomentum(lookback_months=-1)
    
    def test_vol_window_validation(self):
        """Test vol_window must be >= lookback_months."""
        with pytest.raises(ValueError, match="vol_window"):
            TimeSeriesMomentum(lookback_months=12, vol_window=6)
    
    def test_generate_returns_dataframe(self, monthly_prices):
        """Test generate() returns DataFrame."""
        signal = TimeSeriesMomentum(lookback_months=12)
        result = signal.generate(monthly_prices)
        assert isinstance(result, pd.DataFrame)
    
    def test_generate_adds_required_columns(self, monthly_prices):
        """Test generate() adds Signal and Momentum columns."""
        signal = TimeSeriesMomentum(lookback_months=12)
        result = signal.generate(monthly_prices)
        
        assert 'Signal' in result.columns
        assert 'Momentum_12M' in result.columns
        assert 'Volatility' in result.columns
    
    def test_signal_is_continuous(self, monthly_prices):
        """Test signal is continuous (not binary 0/1)."""
        signal = TimeSeriesMomentum(lookback_months=12)
        result = signal.generate(monthly_prices)
        
        # Signal should have wide range of values
        signal_values = result['Signal'].dropna()
        assert len(signal_values.unique()) > 10  # Not just 0/1/-1
    
    def test_volatility_adjustment(self, monthly_prices):
        """Test volatility-adjusted signals differ from raw."""
        signal_vol = TimeSeriesMomentum(lookback_months=12, vol_adjust=True)
        signal_raw = TimeSeriesMomentum(lookback_months=12, vol_adjust=False)
        
        result_vol = signal_vol.generate(monthly_prices)
        result_raw = signal_raw.generate(monthly_prices)
        
        # Signals should differ
        assert not result_vol['Signal'].equals(result_raw['Signal'])
    
    def test_multiple_lookbacks(self, monthly_prices):
        """Test ensemble with multiple lookback periods."""
        signal = TimeSeriesMomentum(
            lookback_months=12,
            additional_lookbacks=[3, 6]
        )
        result = signal.generate(monthly_prices)
        
        assert 'Momentum_3M' in result.columns
        assert 'Momentum_6M' in result.columns
        assert 'Momentum_12M' in result.columns
    
    def test_warmup_period_handling(self, monthly_prices):
        """Test warm-up period signals are NaN."""
        signal = TimeSeriesMomentum(lookback_months=12, vol_window=36)
        result = signal.generate(monthly_prices)
        
        # First 36 rows should have NaN signals
        assert result['Signal'].iloc[:36].isna().all()
        
        # Later rows should have valid signals
        assert result['Signal'].iloc[40:].notna().any()


class TestCrossSectionalMomentum:
    """Test suite for CrossSectionalMomentum."""
    
    @pytest.fixture
    def monthly_prices(self):
        """Generate monthly price data."""
        dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='MS')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.005, 0.02, len(dates))))
        
        return pd.DataFrame({'Close': prices}, index=dates)
    
    def test_initialization(self):
        """Test signal initialization."""
        signal = CrossSectionalMomentum(lookback_months=12, normalization='zscore')
        assert signal.lookback_months == 12
        assert signal.normalization == 'zscore'
    
    def test_invalid_normalization_raises_error(self):
        """Test invalid normalization method raises error."""
        with pytest.raises(ValueError, match="normalization must be"):
            CrossSectionalMomentum(normalization='invalid')
    
    def test_generate_returns_signal(self, monthly_prices):
        """Test signal generation."""
        signal = CrossSectionalMomentum(lookback_months=12)
        result = signal.generate(monthly_prices)
        
        assert 'Signal' in result.columns
        assert 'Momentum_12M' in result.columns
    
    def test_volatility_adjustment(self, monthly_prices):
        """Test vol-adjusted vs. raw signals."""
        signal_vol = CrossSectionalMomentum(lookback_months=12, vol_adjust=True)
        signal_raw = CrossSectionalMomentum(lookback_months=12, vol_adjust=False)
        
        result_vol = signal_vol.generate(monthly_prices)
        result_raw = signal_raw.generate(monthly_prices)
        
        # Should differ when vol_adjust is different
        assert not result_vol['Signal'].equals(result_raw['Signal'])


class TestRiskAdjustedMomentum:
    """Test suite for RiskAdjustedMomentum."""
    
    @pytest.fixture
    def monthly_prices(self):
        """Generate monthly price data."""
        dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='MS')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.005, 0.02, len(dates))))
        
        return pd.DataFrame({'Close': prices}, index=dates)
    
    def test_sharpe_ratio_calculation(self, monthly_prices):
        """Test Sharpe ratio calculation."""
        signal = RiskAdjustedMomentum(lookback_months=12, risk_metric='sharpe')
        result = signal.generate(monthly_prices)
        
        assert 'Signal' in result.columns
        assert 'SharpeRatio' in result.columns
        
        # Sharpe ratios should be reasonable (-3 to 3 typically)
        sharpe_values = result['SharpeRatio'].dropna()
        assert sharpe_values.abs().max() < 10
    
    def test_sortino_ratio_calculation(self, monthly_prices):
        """Test Sortino ratio calculation."""
        signal = RiskAdjustedMomentum(lookback_months=12, risk_metric='sortino')
        result = signal.generate(monthly_prices)
        
        assert 'SortinoRatio' in result.columns
    
    def test_invalid_risk_metric_raises_error(self):
        """Test invalid risk metric raises error."""
        with pytest.raises(ValueError, match="risk_metric must be"):
            RiskAdjustedMomentum(risk_metric='invalid')


class TestYieldCarry:
    """Test suite for YieldCarry signal."""
    
    @pytest.fixture
    def bond_data(self):
        """Generate bond data with yields."""
        dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='MS')
        
        # Simulated yields (2-5% range)
        np.random.seed(42)
        yields = 3.0 + np.cumsum(np.random.normal(0, 0.1, len(dates)))
        yields = np.clip(yields, 0.5, 6.0)
        
        return pd.DataFrame({
            'Close': 100 - yields,  # Bond price (inverse of yield)
            'Yield': yields
        }, index=dates)
    
    @pytest.fixture
    def stock_data(self):
        """Generate stock data with dividends."""
        dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='MS')
        np.random.seed(42)
        
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.005, 0.02, len(dates))))
        dividends = prices * 0.02 / 12  # 2% annual yield
        
        return pd.DataFrame({
            'Close': prices,
            'Dividend': dividends
        }, index=dates)
    
    def test_initialization(self):
        """Test signal initialization."""
        signal = YieldCarry(yield_type='spread', risk_free_rate=0.02)
        assert signal.yield_type == 'spread'
        assert signal.risk_free_rate == 0.02
    
    def test_invalid_yield_type_raises_error(self):
        """Test invalid yield_type raises error."""
        with pytest.raises(ValueError, match="yield_type must be"):
            YieldCarry(yield_type='invalid')
    
    def test_absolute_yield_signal(self, bond_data):
        """Test absolute yield signal."""
        signal = YieldCarry(yield_type='absolute')
        result = signal.generate(bond_data)
        
        assert 'Signal' in result.columns
        
        # Signal should be close to yield/100
        assert np.isclose(
            result['Signal'].iloc[-1],
            bond_data['Yield'].iloc[-1] / 100,
            atol=0.01
        )
    
    def test_spread_yield_signal(self, bond_data):
        """Test yield spread signal."""
        signal = YieldCarry(yield_type='spread', risk_free_rate=0.02)
        result = signal.generate(bond_data)
        
        # Signal should be yield - risk_free
        expected = (bond_data['Yield'] / 100) - 0.02
        assert np.isclose(result['Signal'].iloc[-1], expected.iloc[-1], atol=0.01)
    
    def test_dividend_yield_calculation(self, stock_data):
        """Test dividend yield calculation from Close + Dividend."""
        signal = YieldCarry(yield_type='absolute')
        result = signal.generate(stock_data)
        
        assert 'Signal' in result.columns
        assert result['Signal'].notna().any()
    
    def test_missing_yield_raises_error(self):
        """Test missing yield data raises error."""
        df = pd.DataFrame({'Close': [100, 101, 102]})
        signal = YieldCarry()
        
        with pytest.raises(ValueError, match="must have 'Yield' column"):
            signal.generate(df)


class TestRollYield:
    """Test suite for RollYield signal."""
    
    @pytest.fixture
    def futures_data(self):
        """Generate futures term structure data."""
        dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='MS')
        np.random.seed(42)
        
        # Front month prices
        front = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, len(dates))))
        
        # Next month in contango (+1%) or backwardation (-1%)
        term_structure = np.random.choice([-0.01, 0.01], size=len(dates))
        next_month = front * (1 + term_structure)
        
        return pd.DataFrame({
            'FrontMonth': front,
            'NextMonth': next_month
        }, index=dates)
    
    def test_initialization(self):
        """Test signal initialization."""
        signal = RollYield(term_structure_window=3, annualize=True)
        assert signal.term_structure_window == 3
        assert signal.annualize is True
    
    def test_generate_roll_yield(self, futures_data):
        """Test roll yield calculation."""
        signal = RollYield(term_structure_window=3, annualize=True)
        result = signal.generate(futures_data)
        
        assert 'Signal' in result.columns
        assert 'RollYield' in result.columns
        assert 'TermStructure' in result.columns
    
    def test_missing_columns_raises_error(self):
        """Test missing futures columns raises error."""
        df = pd.DataFrame({'Close': [100, 101, 102]})
        signal = RollYield()
        
        with pytest.raises(ValueError, match="must have 'FrontMonth' and 'NextMonth'"):
            signal.generate(df)


class TestTAAEnsembleSignal:
    """Test suite for TAAEnsembleSignal."""
    
    @pytest.fixture
    def monthly_data(self):
        """Generate data with both price and yield."""
        dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='MS')
        np.random.seed(42)
        
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.005, 0.02, len(dates))))
        yields = 3.0 + np.cumsum(np.random.normal(0, 0.1, len(dates)))
        yields = np.clip(yields, 0.5, 6.0)
        
        return pd.DataFrame({
            'Close': prices,
            'Yield': yields
        }, index=dates)
    
    def test_initialization(self):
        """Test ensemble initialization."""
        ensemble = TAAEnsembleSignal(
            signals={
                'momentum': TimeSeriesMomentum(lookback_months=12),
                'carry': YieldCarry(yield_type='spread')
            },
            weights={'momentum': 0.6, 'carry': 0.4}
        )
        
        assert len(ensemble.signals) == 2
        assert ensemble.weights['momentum'] == 0.6
    
    def test_empty_signals_raises_error(self):
        """Test empty signals dict raises error."""
        with pytest.raises(ValueError, match="Must provide at least one signal"):
            TAAEnsembleSignal(signals={})
    
    def test_weight_mismatch_raises_error(self):
        """Test weight/signal key mismatch raises error."""
        with pytest.raises(ValueError, match="Weight keys must match"):
            TAAEnsembleSignal(
                signals={'momentum': TimeSeriesMomentum()},
                weights={'carry': 1.0}
            )
    
    def test_weights_sum_validation(self):
        """Test weights must sum to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            TAAEnsembleSignal(
                signals={'momentum': TimeSeriesMomentum()},
                weights={'momentum': 0.5}  # Doesn't sum to 1.0
            )
    
    def test_generate_ensemble_signal(self, monthly_data):
        """Test ensemble signal generation."""
        ensemble = TAAEnsembleSignal(
            signals={
                'momentum': TimeSeriesMomentum(lookback_months=12),
                'carry': YieldCarry(yield_type='spread')
            },
            weights={'momentum': 0.6, 'carry': 0.4}
        )
        
        result = ensemble.generate(monthly_data)
        
        assert 'Signal' in result.columns
        assert 'Signal_momentum' in result.columns
        assert 'Signal_carry' in result.columns
    
    def test_equal_weights_default(self, monthly_data):
        """Test default equal weights."""
        ensemble = TAAEnsembleSignal(
            signals={
                'momentum': TimeSeriesMomentum(lookback_months=12),
                'carry': YieldCarry(yield_type='spread')
            }
            # No weights specified
        )
        
        result = ensemble.generate(monthly_data)
        assert 'Signal' in result.columns


class TestMultiAssetEnsemble:
    """Test suite for MultiAssetEnsemble."""
    
    @pytest.fixture
    def multi_asset_prices(self):
        """Generate price data for multiple assets."""
        dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='MS')
        np.random.seed(42)
        
        prices = {}
        for ticker in ['SPY', 'TLT', 'GLD']:
            price_series = 100 * np.exp(np.cumsum(np.random.normal(0.005, 0.02, len(dates))))
            prices[ticker] = pd.DataFrame({
                'Close': price_series,
                'Yield': 2.0 + np.random.normal(0, 0.5, len(dates))
            }, index=dates)
        
        return prices
    
    def test_initialization(self):
        """Test multi-asset ensemble initialization."""
        ensemble = MultiAssetEnsemble(
            signal_generators={
                'momentum': TimeSeriesMomentum(lookback_months=12),
                'carry': YieldCarry(yield_type='spread')
            },
            weights={'momentum': 0.7, 'carry': 0.3}
        )
        
        assert len(ensemble.signal_generators) == 2
        assert ensemble.cross_sectional_norm is True
    
    def test_generate_multi_asset_signals(self, multi_asset_prices):
        """Test multi-asset signal generation."""
        ensemble = MultiAssetEnsemble(
            signal_generators={
                'momentum': TimeSeriesMomentum(lookback_months=12)
            }
        )
        
        results = ensemble.generate_multi_asset(multi_asset_prices)
        
        # Should return dict with all tickers
        assert set(results.keys()) == {'SPY', 'TLT', 'GLD'}
        
        # Each ticker should have Signal column
        for ticker, df in results.items():
            assert 'Signal' in df.columns
            assert 'Signal_momentum' in df.columns
    
    def test_cross_sectional_normalization(self, multi_asset_prices):
        """Test cross-sectional normalization applied."""
        ensemble = MultiAssetEnsemble(
            signal_generators={
                'momentum': TimeSeriesMomentum(lookback_months=12)
            },
            cross_sectional_norm=True
        )
        
        results = ensemble.generate_multi_asset(multi_asset_prices)
        
        # Extract signals at a common date
        signals_at_date = {
            ticker: df.loc['2020-01-01', 'Signal']
            for ticker, df in results.items()
            if '2020-01-01' in df.index
        }
        
        # Cross-sectional normalized signals should have mean ~0
        if len(signals_at_date) > 1:
            mean_signal = np.mean(list(signals_at_date.values()))
            assert abs(mean_signal) < 1.0  # Should be close to 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
