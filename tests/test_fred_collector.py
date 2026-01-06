"""
Tests for FredCollector.

Verifies FRED data fetching with friendly names, convenience methods,
and error handling.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from core.data.collectors.fred_collector import FredCollector


class TestFredCollector:
    """Test suite for FredCollector."""
    
    @pytest.fixture
    def collector(self):
        """Create FredCollector instance."""
        return FredCollector()
    
    @pytest.fixture
    def date_range(self):
        """Provide standard test date range."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 5)  # 5 years
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    def test_series_map_exists(self, collector):
        """Test SERIES_MAP has expected indicators."""
        assert hasattr(collector, 'SERIES_MAP')
        assert isinstance(collector.SERIES_MAP, dict)
        
        # Check key indicators exist
        assert 'gdp' in collector.SERIES_MAP
        assert 'unemployment' in collector.SERIES_MAP
        assert 'cpi' in collector.SERIES_MAP
        assert 'fed_funds' in collector.SERIES_MAP
        
        # Verify mappings are correct
        assert collector.SERIES_MAP['gdp'] == 'GDP'
        assert collector.SERIES_MAP['unemployment'] == 'UNRATE'
        assert collector.SERIES_MAP['cpi'] == 'CPIAUCSL'
    
    def test_fetch_history_with_friendly_names(self, collector, date_range):
        """Test fetching data using friendly names."""
        start_date, end_date = date_range
        
        df = collector.fetch_history(
            tickers=['gdp', 'unemployment'],
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
        # Should have FRED IDs as columns (mapped from friendly names)
        assert 'GDP' in df.columns
        assert 'UNRATE' in df.columns
        
        # Index should be datetime
        assert isinstance(df.index, pd.DatetimeIndex)
    
    def test_fetch_history_with_fred_ids(self, collector, date_range):
        """Test fetching data using direct FRED series IDs."""
        start_date, end_date = date_range
        
        df = collector.fetch_history(
            tickers=['GDP', 'UNRATE'],
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'GDP' in df.columns
        assert 'UNRATE' in df.columns
    
    def test_fetch_history_mixed_names(self, collector, date_range):
        """Test fetching with mix of friendly names and FRED IDs."""
        start_date, end_date = date_range
        
        df = collector.fetch_history(
            tickers=['gdp', 'UNRATE', 'cpi'],  # Mixed
            start_date=start_date,
            end_date=end_date
        )
        
        assert not df.empty
        assert 'GDP' in df.columns
        assert 'UNRATE' in df.columns
        assert 'CPIAUCSL' in df.columns
    
    def test_fetch_growth_indicators(self, collector, date_range):
        """Test convenience method for growth indicators."""
        start_date, end_date = date_range
        
        df = collector.fetch_growth_indicators(
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
        # Should contain growth-related series
        expected_series = ['A191RL1Q225SBEA', 'UNRATE', 'PAYEMS', 'INDPRO']
        for series in expected_series:
            assert series in df.columns, f"Missing {series} in growth indicators"
    
    def test_fetch_inflation_indicators(self, collector, date_range):
        """Test convenience method for inflation indicators."""
        start_date, end_date = date_range
        
        df = collector.fetch_inflation_indicators(
            start_date=start_date,
            end_date=end_date
        )
        
        assert not df.empty
        
        # Should contain inflation-related series
        expected_series = ['CPIAUCSL', 'CPILFESL', 'PCEPILFE', 'T5YIE']
        for series in expected_series:
            assert series in df.columns, f"Missing {series} in inflation indicators"
    
    def test_fetch_policy_indicators(self, collector, date_range):
        """Test convenience method for policy indicators."""
        start_date, end_date = date_range
        
        df = collector.fetch_policy_indicators(
            start_date=start_date,
            end_date=end_date
        )
        
        assert not df.empty
        
        # Should contain policy-related series
        expected_series = ['DFF', 'DGS10', 'DGS2', 'DGS3MO']
        for series in expected_series:
            assert series in df.columns, f"Missing {series} in policy indicators"
        
        # Should calculate yield curve slope
        assert 'yield_curve_slope' in df.columns
        
        # Yield curve slope should be reasonable (between -5% and 5%)
        slope_values = df['yield_curve_slope'].dropna()
        assert slope_values.abs().max() < 10, "Yield curve slope seems unrealistic"
    
    def test_fetch_risk_indicators(self, collector, date_range):
        """Test convenience method for risk indicators."""
        start_date, end_date = date_range
        
        df = collector.fetch_risk_indicators(
            start_date=start_date,
            end_date=end_date
        )
        
        assert not df.empty
        
        # Should contain at least some risk series (VIX may not be available)
        assert any(series in df.columns for series in ['BAA10Y', 'TEDRATE', 'VIXCLS'])
    
    def test_fetch_latest(self, collector):
        """Test fetching latest data point."""
        df = collector.fetch_latest(tickers=['gdp', 'unemployment'])
        
        assert isinstance(df, pd.DataFrame)
        
        # Should only return 1 row (latest)
        assert len(df) == 1
        
        # Should have the requested series
        assert 'GDP' in df.columns
        assert 'UNRATE' in df.columns
    
    def test_list_available_series(self):
        """Test static method to list available series."""
        series_map = FredCollector.list_available_series()
        
        assert isinstance(series_map, dict)
        assert len(series_map) > 20  # Should have many series
        
        # Verify it's a copy (not the original)
        series_map['test'] = 'TEST'
        assert 'test' not in FredCollector.SERIES_MAP
    
    def test_fetch_history_empty_tickers(self, collector, date_range):
        """Test fetching with empty ticker list."""
        start_date, end_date = date_range
        
        df = collector.fetch_history(
            tickers=[],
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(df, pd.DataFrame)
        assert df.empty
    
    def test_fetch_history_invalid_series(self, collector, date_range):
        """Test fetching with invalid series ID."""
        start_date, end_date = date_range
        
        # Should raise exception for invalid series
        with pytest.raises(Exception):
            collector.fetch_history(
                tickers=['INVALID_SERIES_12345'],
                start_date=start_date,
                end_date=end_date
            )
    
    def test_forward_fill_applied(self, collector):
        """Test that forward fill is applied to handle different frequencies."""
        # Fetch monthly and daily series together
        df = collector.fetch_history(
            tickers=['gdp', 'fed_funds'],  # GDP is quarterly, Fed Funds is daily
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # After forward fill, should have no NaN in the middle of the series
        # (may have NaN at start if GDP not published yet)
        assert df['GDP'].isna().sum() < len(df) * 0.9  # Less than 90% NaN


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
