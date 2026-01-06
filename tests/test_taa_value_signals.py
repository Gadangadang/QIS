"""
Tests for TAA Value Signal Generators.

Verifies valuation-based signals for equities, bonds, and commodities.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from signals.taa.value import (
    CAPESignal,
    RealYieldSpread,
    EarningsYieldSignal,
    RelativeValueSignal,
    CommodityValueSignal,
    CrossSectionalValue
)


class TestCAPESignal:
    """Test suite for CAPESignal."""
    
    @pytest.fixture
    def cape_data(self):
        """Generate monthly data with CAPE ratios."""
        dates = pd.date_range(start='2000-01-01', end='2023-12-31', freq='MS')
        np.random.seed(42)
        
        # Simulated CAPE ratios (15-35 range, cyclical)
        cape = 20 + 10 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 2, len(dates))
        cape = np.clip(cape, 10, 40)
        
        return pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.normal(0.5, 2, len(dates))),
            'CAPE': cape
        }, index=dates)
    
    def test_initialization(self):
        """Test signal initialization."""
        signal = CAPESignal(normalization='zscore', lookback_years=20)
        assert signal.normalization == 'zscore'
        assert signal.lookback_years == 20
        assert signal.invert is True
    
    def test_invalid_normalization_raises_error(self):
        """Test invalid normalization method raises error."""
        with pytest.raises(ValueError, match="normalization must be"):
            CAPESignal(normalization='invalid')
    
    def test_missing_cape_column_raises_error(self):
        """Test missing CAPE column raises error."""
        df = pd.DataFrame({'Close': [100, 101, 102]})
        signal = CAPESignal()
        
        with pytest.raises(ValueError, match="must have 'CAPE' column"):
            signal.generate(df)
    
    def test_zscore_normalization(self, cape_data):
        """Test z-score normalization."""
        signal = CAPESignal(normalization='zscore', lookback_years=10, invert=True)
        result = signal.generate(cape_data)
        
        assert 'Signal' in result.columns
        assert 'CAPE_ZScore' in result.columns
        
        # Signal should be inverted z-score (high CAPE = negative signal)
        valid_signals = result['Signal'].dropna()
        assert len(valid_signals) > 0
        
        # Z-scores should be roughly between -3 and 3
        assert result['CAPE_ZScore'].dropna().abs().max() < 5
    
    def test_percentile_normalization(self, cape_data):
        """Test percentile normalization."""
        signal = CAPESignal(normalization='percentile', lookback_years=10, invert=True)
        result = signal.generate(cape_data)
        
        assert 'CAPE_Percentile' in result.columns
        
        # Percentiles should be between 0 and 1
        percentiles = result['CAPE_Percentile'].dropna()
        assert (percentiles >= 0).all() and (percentiles <= 1).all()
        
        # Signal should be 1 - percentile when inverted
        valid_idx = result['Signal'].notna()
        assert np.allclose(
            result.loc[valid_idx, 'Signal'],
            1 - result.loc[valid_idx, 'CAPE_Percentile'],
            rtol=0.01
        )
    
    def test_spread_normalization(self, cape_data):
        """Test spread normalization."""
        signal = CAPESignal(normalization='spread', lookback_years=10, invert=True)
        result = signal.generate(cape_data)
        
        assert 'CAPE_Spread' in result.columns
        
        # Spread should be reasonable (within -50% to +50%)
        spreads = result['CAPE_Spread'].dropna()
        assert spreads.abs().max() < 1.0
    
    def test_warmup_period(self, cape_data):
        """Test warm-up period is NaN."""
        signal = CAPESignal(normalization='zscore', lookback_years=10)
        result = signal.generate(cape_data)
        
        # First 10 years should be NaN
        warmup = 10 * 12
        assert result['Signal'].iloc[:warmup].isna().all()
        
        # Later should have values
        assert result['Signal'].iloc[warmup+10:].notna().any()


class TestRealYieldSpread:
    """Test suite for RealYieldSpread."""
    
    @pytest.fixture
    def bond_data(self):
        """Generate bond yield data."""
        dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='MS')
        np.random.seed(42)
        
        nominal_yield = 3.0 + np.cumsum(np.random.normal(0, 0.1, len(dates)))
        nominal_yield = np.clip(nominal_yield, 0.5, 6.0)
        
        inflation = 2.0 + np.cumsum(np.random.normal(0, 0.05, len(dates)))
        inflation = np.clip(inflation, 0, 5.0)
        
        return pd.DataFrame({
            'NominalYield': nominal_yield,
            'InflationExpectation': inflation
        }, index=dates)
    
    def test_initialization(self):
        """Test signal initialization."""
        signal = RealYieldSpread(comparison='historical', lookback_years=10)
        assert signal.comparison == 'historical'
        assert signal.lookback_years == 10
    
    def test_invalid_comparison_raises_error(self):
        """Test invalid comparison method raises error."""
        with pytest.raises(ValueError, match="comparison must be"):
            RealYieldSpread(comparison='invalid')
    
    def test_real_yield_calculation(self, bond_data):
        """Test real yield is calculated correctly."""
        signal = RealYieldSpread(comparison='historical')
        result = signal.generate(bond_data)
        
        assert 'RealYield' in result.columns
        
        # Real yield should be nominal - inflation
        expected = bond_data['NominalYield'] - bond_data['InflationExpectation']
        assert np.allclose(result['RealYield'], expected, rtol=0.01)
    
    def test_historical_comparison(self, bond_data):
        """Test historical comparison generates z-scores."""
        signal = RealYieldSpread(comparison='historical', lookback_years=5)
        result = signal.generate(bond_data)
        
        assert 'Signal' in result.columns
        
        # Signal should be z-score of real yield
        valid_signals = result['Signal'].dropna()
        assert len(valid_signals) > 0
        assert valid_signals.abs().max() < 5
    
    def test_cross_sectional_comparison(self, bond_data):
        """Test cross-sectional returns real yield."""
        signal = RealYieldSpread(comparison='cross_sectional')
        result = signal.generate(bond_data)
        
        # Signal should equal real yield for cross-sectional
        assert 'Signal' in result.columns
        valid_idx = result['Signal'].notna() & result['RealYield'].notna()
        assert np.allclose(
            result.loc[valid_idx, 'Signal'],
            result.loc[valid_idx, 'RealYield'],
            rtol=0.01
        )
    
    def test_missing_yield_columns_raises_error(self):
        """Test missing required columns raises error."""
        df = pd.DataFrame({'Close': [100, 101, 102]})
        signal = RealYieldSpread()
        
        with pytest.raises(ValueError, match="must have"):
            signal.generate(df)


class TestEarningsYieldSignal:
    """Test suite for EarningsYieldSignal."""
    
    @pytest.fixture
    def equity_data(self):
        """Generate equity data with earnings."""
        dates = pd.date_range(start='2005-01-01', end='2023-12-31', freq='MS')
        np.random.seed(42)
        
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.005, 0.02, len(dates))))
        pe_ratio = 15 + 5 * np.sin(np.linspace(0, 3*np.pi, len(dates))) + np.random.normal(0, 2, len(dates))
        pe_ratio = np.clip(pe_ratio, 8, 30)
        
        return pd.DataFrame({
            'Price': prices,
            'PE_Ratio': pe_ratio,
            'BondYield': 3.0 + np.random.normal(0, 0.5, len(dates))
        }, index=dates)
    
    def test_initialization(self):
        """Test signal initialization."""
        signal = EarningsYieldSignal(comparison='zscore', lookback_years=15)
        assert signal.comparison == 'zscore'
        assert signal.lookback_years == 15
    
    def test_invalid_comparison_raises_error(self):
        """Test invalid comparison raises error."""
        with pytest.raises(ValueError, match="comparison must be"):
            EarningsYieldSignal(comparison='invalid')
    
    def test_earnings_yield_from_pe(self, equity_data):
        """Test earnings yield calculated from P/E ratio."""
        signal = EarningsYieldSignal(comparison='absolute')
        result = signal.generate(equity_data)
        
        assert 'EarningsYield' in result.columns
        
        # E/Y = 1 / P/E
        expected = 1 / equity_data['PE_Ratio']
        assert np.allclose(result['EarningsYield'], expected, rtol=0.01)
    
    def test_absolute_comparison(self, equity_data):
        """Test absolute earnings yield signal."""
        signal = EarningsYieldSignal(comparison='absolute')
        result = signal.generate(equity_data)
        
        # Signal should equal earnings yield
        assert 'Signal' in result.columns
        assert np.allclose(result['Signal'], result['EarningsYield'], rtol=0.01)
    
    def test_vs_bonds_comparison(self, equity_data):
        """Test earnings yield vs bonds (equity risk premium)."""
        signal = EarningsYieldSignal(comparison='vs_bonds')
        result = signal.generate(equity_data)
        
        assert 'EquityRiskPremium' in result.columns
        
        # ERP = E/Y - Bond Yield
        expected = result['EarningsYield'] - equity_data['BondYield']
        assert np.allclose(result['EquityRiskPremium'], expected, rtol=0.01)
    
    def test_zscore_comparison(self, equity_data):
        """Test z-score comparison."""
        signal = EarningsYieldSignal(comparison='zscore', lookback_years=10)
        result = signal.generate(equity_data)
        
        # Signal should be z-score
        valid_signals = result['Signal'].dropna()
        assert len(valid_signals) > 0
        assert valid_signals.abs().max() < 5


class TestRelativeValueSignal:
    """Test suite for RelativeValueSignal."""
    
    @pytest.fixture
    def valuation_data(self):
        """Generate data with P/B ratios."""
        dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='MS')
        np.random.seed(42)
        
        pb_ratio = 2.0 + 1.0 * np.sin(np.linspace(0, 3*np.pi, len(dates))) + np.random.normal(0, 0.3, len(dates))
        pb_ratio = np.clip(pb_ratio, 0.5, 5.0)
        
        return pd.DataFrame({
            'PB_Ratio': pb_ratio,
            'DivYield': 2.0 + np.random.normal(0, 0.5, len(dates))
        }, index=dates)
    
    def test_initialization(self):
        """Test signal initialization."""
        signal = RelativeValueSignal(metric_column='PB_Ratio', invert=True)
        assert signal.metric_column == 'PB_Ratio'
        assert signal.invert is True
    
    def test_missing_metric_column_raises_error(self):
        """Test missing metric column raises error."""
        df = pd.DataFrame({'Close': [100, 101, 102]})
        signal = RelativeValueSignal(metric_column='PE_Ratio')
        
        with pytest.raises(ValueError, match="must have 'PE_Ratio' column"):
            signal.generate(df)
    
    def test_inverted_signal(self, valuation_data):
        """Test inverted signal (high P/B = negative)."""
        signal = RelativeValueSignal(metric_column='PB_Ratio', invert=True, lookback_years=5)
        result = signal.generate(valuation_data)
        
        assert 'Signal' in result.columns
        assert 'Value_ZScore' in result.columns
        
        # Signal should be negative z-score
        valid_idx = result['Signal'].notna() & result['Value_ZScore'].notna()
        assert np.allclose(
            result.loc[valid_idx, 'Signal'],
            -result.loc[valid_idx, 'Value_ZScore'],
            rtol=0.01
        )
    
    def test_non_inverted_signal(self, valuation_data):
        """Test non-inverted signal (high div yield = positive)."""
        signal = RelativeValueSignal(metric_column='DivYield', invert=False, lookback_years=5)
        result = signal.generate(valuation_data)
        
        # Signal should equal z-score (not inverted)
        valid_idx = result['Signal'].notna() & result['Value_ZScore'].notna()
        assert np.allclose(
            result.loc[valid_idx, 'Signal'],
            result.loc[valid_idx, 'Value_ZScore'],
            rtol=0.01
        )
    
    def test_outlier_clipping(self, valuation_data):
        """Test outlier clipping works."""
        signal = RelativeValueSignal(metric_column='PB_Ratio', outlier_clip=2.0)
        result = signal.generate(valuation_data)
        
        # Z-scores should be clipped at ±2.0
        zscores = result['Value_ZScore'].dropna()
        assert zscores.max() <= 2.0
        assert zscores.min() >= -2.0


class TestCommodityValueSignal:
    """Test suite for CommodityValueSignal."""
    
    @pytest.fixture
    def commodity_data(self):
        """Generate commodity data."""
        dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='MS')
        np.random.seed(42)
        
        prices = 50 * np.exp(np.cumsum(np.random.normal(0, 0.03, len(dates))))
        inventory = 100 + 20 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 5, len(dates))
        
        return pd.DataFrame({
            'Close': prices,
            'Price': prices,
            'Inventory': inventory,
            'MarginalCost': prices * 0.7
        }, index=dates)
    
    def test_initialization(self):
        """Test signal initialization."""
        signal = CommodityValueSignal(method='price_trend', lookback_years=5)
        assert signal.method == 'price_trend'
        assert signal.lookback_years == 5
    
    def test_invalid_method_raises_error(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="method must be"):
            CommodityValueSignal(method='invalid')
    
    def test_price_trend_method(self, commodity_data):
        """Test price trend method."""
        signal = CommodityValueSignal(method='price_trend', lookback_years=3)
        result = signal.generate(commodity_data)
        
        assert 'Signal' in result.columns
        
        # Signal should be inverted z-score (low price = positive)
        valid_signals = result['Signal'].dropna()
        assert len(valid_signals) > 0
    
    def test_inventory_method(self, commodity_data):
        """Test inventory method."""
        signal = CommodityValueSignal(method='inventory', lookback_years=3)
        result = signal.generate(commodity_data)
        
        assert 'Signal' in result.columns
        valid_signals = result['Signal'].dropna()
        assert len(valid_signals) > 0
    
    def test_cost_spread_method(self, commodity_data):
        """Test cost spread method."""
        signal = CommodityValueSignal(method='cost_spread', lookback_years=3)
        result = signal.generate(commodity_data)
        
        assert 'Signal' in result.columns
        assert 'CostSpread' in result.columns
        
        # Cost spread should be reasonable
        spreads = result['CostSpread'].dropna()
        assert spreads.abs().max() < 2.0


class TestCrossSectionalValue:
    """Test suite for CrossSectionalValue."""
    
    @pytest.fixture
    def multi_asset_data(self):
        """Generate multi-asset valuation data."""
        dates = pd.date_range(start='2015-01-01', end='2023-12-31', freq='MS')
        np.random.seed(42)
        
        return pd.DataFrame({
            'PE_Ratio': 15 + np.random.normal(0, 3, len(dates)),
            'PB_Ratio': 2.0 + np.random.normal(0, 0.5, len(dates)),
            'DivYield': 2.5 + np.random.normal(0, 0.5, len(dates))
        }, index=dates)
    
    def test_initialization(self):
        """Test signal initialization."""
        signal = CrossSectionalValue(metric_column='PE_Ratio', invert_rank=True)
        assert signal.metric_column == 'PE_Ratio'
        assert signal.invert_rank is True
    
    def test_inverted_rank(self, multi_asset_data):
        """Test inverted rank for P/E."""
        signal = CrossSectionalValue(metric_column='PE_Ratio', invert_rank=True)
        result = signal.generate(multi_asset_data)
        
        assert 'Signal' in result.columns
        
        # Signal should be negative PE (low PE = positive signal)
        assert np.allclose(result['Signal'], -multi_asset_data['PE_Ratio'], rtol=0.01)
    
    def test_non_inverted_rank(self, multi_asset_data):
        """Test non-inverted rank for div yield."""
        signal = CrossSectionalValue(metric_column='DivYield', invert_rank=False)
        result = signal.generate(multi_asset_data)
        
        # Signal should equal metric (high div yield = positive)
        assert np.allclose(result['Signal'], multi_asset_data['DivYield'], rtol=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
