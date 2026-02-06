"""
Simple tests for uncovered asset_registry functions.

Tests for core/asset_registry.py utility functions.
"""

import pytest
from io import StringIO
import sys
from core.asset_registry import (
    print_registry_summary,
    get_futures_requiring_rollover,
    get_seasonal_commodities,
    filter_by_class,
    AssetClass
)


def test_print_registry_summary():
    """Test print_registry_summary outputs without error."""
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        print_registry_summary()
        output = sys.stdout.getvalue()
        
        # Check that summary includes expected headers
        assert "ASSET REGISTRY SUMMARY" in output
        assert "By Asset Type:" in output
        assert "By Asset Class:" in output
        assert "Total Assets:" in output
        
    finally:
        sys.stdout = old_stdout


def test_get_futures_requiring_rollover():
    """Test getting futures that require rollover."""
    rollover_assets = get_futures_requiring_rollover()
    
    # Should return a list
    assert isinstance(rollover_assets, list)
    
    # All should have requires_rollover = True
    for asset in rollover_assets:
        assert asset.requires_rollover == True


def test_get_seasonal_commodities():
    """Test getting commodities with seasonal patterns."""
    seasonal = get_seasonal_commodities()
    
    # Should return a list
    assert isinstance(seasonal, list)
    
    # All should have seasonality_pattern set
    for asset in seasonal:
        assert asset.seasonality_pattern is not None
        assert asset.seasonality_pattern != ""


def test_filter_by_class_energy():
    """Test filtering by energy commodity class."""
    energy = filter_by_class(AssetClass.COMMODITY_ENERGY)
    
    # Should have some energy commodities
    assert len(energy) > 0
    
    # All should be COMMODITY_ENERGY class
    for asset in energy:
        assert asset.asset_class == AssetClass.COMMODITY_ENERGY


def test_filter_by_class_metals():
    """Test filtering by metal commodity class."""
    metals = filter_by_class(AssetClass.COMMODITY_METAL)
    
    # Should have some metal commodities
    assert len(metals) > 0
    
    # All should be COMMODITY_METAL class
    for asset in metals:
        assert asset.asset_class == AssetClass.COMMODITY_METAL


def test_filter_by_class_equities():
    """Test filtering by equity class."""
    equities = filter_by_class(AssetClass.EQUITY)
    
    # Should return a list (may be empty)
    assert isinstance(equities, list)


def test_filter_by_class_fixed_income():
    """Test filtering by fixed income class."""
    fixed_income = filter_by_class(AssetClass.FIXED_INCOME)
    
    # Should return a list
    assert isinstance(fixed_income, list)
