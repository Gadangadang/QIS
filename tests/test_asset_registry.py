"""
Unit tests for asset registry functionality.

Tests:
- Asset metadata retrieval
- Registry queries and filters
- Contract specifications
- Asset validation

Run with: pytest tests/test_asset_registry.py -v
"""

import pytest
from core.asset_registry import (
    AssetType, AssetClass, AssetMetadata, 
    ASSET_REGISTRY, get_asset, filter_by_class, get_futures_requiring_rollover
)


class TestAssetMetadata:
    """Test AssetMetadata dataclass."""
    
    def test_create_asset_metadata(self):
        """Test creating asset metadata."""
        metadata = AssetMetadata(
            ticker='TEST',
            name='Test Asset',
            asset_type=AssetType.FUTURES,
            asset_class=AssetClass.EQUITY,
            contract_multiplier=50,
            tick_size=0.25,
            expiration_cycle='quarterly',
            requires_rollover=True,
            yfinance_symbol='TEST=F',
            typical_margin_pct=0.05
        )
        
        assert metadata.ticker == 'TEST'
        assert metadata.contract_multiplier == 50
        assert metadata.requires_rollover is True
    
    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        metadata = AssetMetadata(
            ticker='SPY',
            name='SPDR S&P 500 ETF',
            asset_type=AssetType.ETF,
            asset_class=AssetClass.EQUITY
        )
        
        assert metadata.contract_multiplier is None
        assert metadata.requires_rollover is False


class TestAssetRegistry:
    """Test ASSET_REGISTRY functionality."""
    
    def test_registry_contains_key_assets(self):
        """Test that registry contains expected assets."""
        assert 'ES' in ASSET_REGISTRY
        assert 'NQ' in ASSET_REGISTRY
        assert 'GC' in ASSET_REGISTRY
        assert 'CL' in ASSET_REGISTRY
        assert 'NG' in ASSET_REGISTRY
    
    def test_get_asset_success(self):
        """Test retrieving existing asset."""
        es_metadata = get_asset('ES')
        
        assert es_metadata.ticker == 'ES'
        assert es_metadata.name == 'S&P 500 E-mini Futures'
        assert es_metadata.asset_type == AssetType.FUTURES
        assert es_metadata.contract_multiplier == 50
    
    def test_get_asset_not_found(self):
        """Test that unknown asset raises KeyError."""
        with pytest.raises(KeyError):
            get_asset('INVALID_TICKER')
    
    def test_filter_by_class_equity(self):
        """Test filtering by equity asset class."""
        equity_futures = filter_by_class(AssetClass.EQUITY)
        
        assert len(equity_futures) >= 2  # At least ES and NQ
        tickers = [a.ticker for a in equity_futures]
        assert 'ES' in tickers
        assert 'NQ' in tickers
    
    def test_filter_by_class_energy(self):
        """Test filtering by energy commodity class."""
        energy = filter_by_class(AssetClass.COMMODITY_ENERGY)
        
        tickers = [a.ticker for a in energy]
        assert 'CL' in tickers  # Crude Oil
        assert 'NG' in tickers  # Natural Gas
    
    def test_filter_by_class_metals(self):
        """Test filtering by metals commodity class."""
        metals = filter_by_class(AssetClass.COMMODITY_METAL)
        
        tickers = [a.ticker for a in metals]
        assert 'GC' in tickers  # Gold
    
    def test_requires_rollover(self):
        """Test rollover requirement checking."""
        rollover_assets = get_futures_requiring_rollover()
        rollover_tickers = [a.ticker for a in rollover_assets]
        
        # Futures should require rollover
        assert 'ES' in rollover_tickers
        assert 'CL' in rollover_tickers
        assert 'GC' in rollover_tickers
        
        # ETFs should not
        assert 'SPY' not in rollover_tickers


class TestContractSpecifications:
    """Test that contract specifications are correct."""
    
    def test_es_contract_specs(self):
        """Test ES (S&P 500 E-mini) specifications."""
        es = get_asset('ES')
        
        assert es.contract_multiplier == 50
        assert es.expiration_cycle == 'quarterly'
        assert es.yfinance_symbol == 'ES=F'
    
    def test_cl_contract_specs(self):
        """Test CL (Crude Oil) specifications."""
        cl = get_asset('CL')
        
        assert cl.contract_multiplier == 1000  # 1,000 barrels
        assert cl.asset_class == AssetClass.COMMODITY_ENERGY
        assert cl.expiration_cycle == 'monthly'
    
    def test_ng_contract_specs(self):
        """Test NG (Natural Gas) specifications."""
        ng = get_asset('NG')
        
        assert ng.contract_multiplier == 10000  # 10,000 MMBtu
        assert ng.asset_class == AssetClass.COMMODITY_ENERGY
    
    def test_gc_contract_specs(self):
        """Test GC (Gold) specifications."""
        gc = get_asset('GC')
        
        assert gc.contract_multiplier == 100  # 100 troy ounces
        assert gc.asset_class == AssetClass.COMMODITY_METAL
    
    def test_nq_contract_specs(self):
        """Test NQ (Nasdaq 100 E-mini) specifications."""
        nq = get_asset('NQ')
        
        assert nq.contract_multiplier == 20
        assert nq.asset_class == AssetClass.EQUITY


class TestAssetEnums:
    """Test AssetType and AssetClass enums."""
    
    def test_asset_type_enum(self):
        """Test AssetType enum values."""
        assert AssetType.FUTURES.value == 'futures'
        assert AssetType.STOCK.value == 'stock'
        assert AssetType.ETF.value == 'etf'
        assert AssetType.INDEX.value == 'index'
        assert AssetType.FOREX.value == 'forex'
    
    def test_asset_class_enum(self):
        """Test AssetClass enum values."""
        assert AssetClass.EQUITY.value == 'equity'
        assert AssetClass.COMMODITY_ENERGY.value == 'commodity_energy'
        assert AssetClass.COMMODITY_METAL.value == 'commodity_metal'
        assert AssetClass.COMMODITY_AGRICULTURE.value == 'commodity_agriculture'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
