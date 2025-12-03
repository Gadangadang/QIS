"""
Extended tests for asset registry helper functions.

Tests uncovered functions:
- filter_by_type()
- get_seasonal_commodities()
- get_all_tickers()

Run with: pytest tests/test_asset_registry_extended.py -v
"""

import pytest
from core.asset_registry import (
    AssetType,
    AssetClass,
    filter_by_type,
    get_seasonal_commodities,
    get_all_tickers,
)


class TestFilterByType:
    """Test filter_by_type helper function."""
    
    def test_filter_futures(self):
        """Test filtering by FUTURES type."""
        futures = filter_by_type(AssetType.FUTURES)
        
        # Should have multiple futures
        assert len(futures) > 0
        
        # All should be futures
        for asset in futures:
            assert asset.asset_type == AssetType.FUTURES
        
        # Should include key futures
        tickers = [a.ticker for a in futures]
        assert 'ES' in tickers
        assert 'CL' in tickers
        assert 'GC' in tickers
    
    def test_filter_etfs(self):
        """Test filtering by ETF type."""
        etfs = filter_by_type(AssetType.ETF)
        
        assert len(etfs) > 0
        
        # All should be ETFs
        for asset in etfs:
            assert asset.asset_type == AssetType.ETF
        
        # Should include common ETFs
        tickers = [a.ticker for a in etfs]
        assert 'SPY' in tickers
        assert 'QQQ' in tickers
    
    def test_filter_stocks(self):
        """Test filtering by STOCK type."""
        stocks = filter_by_type(AssetType.STOCK)
        
        # All should be stocks
        for asset in stocks:
            assert asset.asset_type == AssetType.STOCK


class TestSeasonalCommodities:
    """Test get_seasonal_commodities function."""
    
    def test_returns_seasonal_assets(self):
        """Test that seasonal commodities are returned."""
        seasonal = get_seasonal_commodities()
        
        assert len(seasonal) > 0
        
        # All should have seasonality flag
        for asset in seasonal:
            assert asset.has_seasonality is True
            assert asset.seasonality_pattern is not None
    
    def test_includes_energy_seasonals(self):
        """Test that energy commodities with seasonality are included."""
        seasonal = get_seasonal_commodities()
        tickers = [a.ticker for a in seasonal]
        
        # CL has summer driving seasonality
        assert 'CL' in tickers
        
        # NG has winter demand seasonality
        assert 'NG' in tickers
    
    def test_excludes_non_seasonal(self):
        """Test that non-seasonal assets are excluded."""
        seasonal = get_seasonal_commodities()
        tickers = [a.ticker for a in seasonal]
        
        # Gold doesn't have seasonality
        assert 'GC' not in tickers or any(
            a.ticker == 'GC' and a.has_seasonality for a in seasonal
        )


class TestGetAllTickers:
    """Test get_all_tickers function."""
    
    def test_returns_sorted_list(self):
        """Test that tickers are returned in sorted order."""
        tickers = get_all_tickers()
        
        assert len(tickers) > 0
        assert isinstance(tickers, list)
        
        # Should be sorted
        assert tickers == sorted(tickers)
    
    def test_includes_key_assets(self):
        """Test that key assets are included."""
        tickers = get_all_tickers()
        
        # Equity futures
        assert 'ES' in tickers
        assert 'NQ' in tickers
        
        # Energy
        assert 'CL' in tickers
        assert 'NG' in tickers
        
        # Metals
        assert 'GC' in tickers
        
        # ETFs
        assert 'SPY' in tickers
        assert 'QQQ' in tickers
    
    def test_no_duplicates(self):
        """Test that there are no duplicate tickers."""
        tickers = get_all_tickers()
        
        assert len(tickers) == len(set(tickers))


class TestAssetTypeFiltering:
    """Test comprehensive filtering by asset type."""
    
    def test_futures_have_contract_multipliers(self):
        """Test that futures have contract multiplier specified."""
        futures = filter_by_type(AssetType.FUTURES)
        
        for asset in futures:
            assert asset.contract_multiplier is not None
            assert asset.contract_multiplier > 0
            assert asset.requires_rollover is True
    
    def test_etfs_no_rollover(self):
        """Test that ETFs don't require rollover."""
        etfs = filter_by_type(AssetType.ETF)
        
        for asset in etfs:
            assert asset.requires_rollover is False
            assert asset.contract_multiplier is None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
