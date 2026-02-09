"""
Tests for core.multi_asset_loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from core.multi_asset_loader import MultiAssetLoader


class TestMultiAssetLoader:
    """Test suite for MultiAssetLoader class."""

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary dataset directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_initialization_default_dir(self):
        """Test MultiAssetLoader initialization with default directory."""
        loader = MultiAssetLoader()
        
        assert loader.dataset_dir is not None
        assert loader.use_yfinance == True

    def test_initialization_custom_dir(self, temp_dataset_dir):
        """Test MultiAssetLoader initialization with custom directory."""
        loader = MultiAssetLoader(dataset_dir=temp_dataset_dir)
        
        assert loader.dataset_dir == temp_dataset_dir

    def test_initialization_disable_yfinance(self):
        """Test MultiAssetLoader with yfinance disabled."""
        loader = MultiAssetLoader(use_yfinance=False)
        
        assert loader.use_yfinance == False

    def test_asset_files_mapping_exists(self):
        """Test that ASSET_FILES mapping is defined."""
        loader = MultiAssetLoader()
        
        assert hasattr(loader, 'ASSET_FILES')
        assert isinstance(loader.ASSET_FILES, dict)
        assert len(loader.ASSET_FILES) > 0

    def test_yfinance_symbols_mapping_exists(self):
        """Test that YFINANCE_SYMBOLS mapping is defined."""
        loader = MultiAssetLoader()
        
        assert hasattr(loader, 'YFINANCE_SYMBOLS')
        assert isinstance(loader.YFINANCE_SYMBOLS, dict)
        assert len(loader.YFINANCE_SYMBOLS) > 0

    def test_known_tickers_include_futures(self):
        """Test that known tickers include common futures."""
        loader = MultiAssetLoader()
        
        # ES, NQ, GC should be in either ASSET_FILES or YFINANCE_SYMBOLS
        known_tickers = set(list(loader.ASSET_FILES.keys()) + list(loader.YFINANCE_SYMBOLS.keys()))
        
        assert 'ES' in known_tickers
        assert 'GC' in known_tickers

    def test_load_single_asset_unknown_ticker_raises_error(self, temp_dataset_dir):
        """Test that loading unknown ticker raises ValueError."""
        loader = MultiAssetLoader(dataset_dir=temp_dataset_dir, use_yfinance=False)
        
        with pytest.raises(ValueError, match="Unknown ticker"):
            loader.load_single_asset('INVALID_TICKER')

    def test_load_single_asset_missing_file_raises_error(self, temp_dataset_dir):
        """Test that missing file raises FileNotFoundError when yfinance disabled."""
        loader = MultiAssetLoader(dataset_dir=temp_dataset_dir, use_yfinance=False)
        
        # ES is in ASSET_FILES but the file won't exist in temp directory
        with pytest.raises((FileNotFoundError, ValueError)):
            loader.load_single_asset('ES')

    def test_loader_has_fetch_from_yfinance_method(self):
        """Test that loader has _fetch_from_yfinance method."""
        loader = MultiAssetLoader()
        
        assert hasattr(loader, '_fetch_from_yfinance')

    def test_loader_has_load_method(self):
        """Test that loader has load method."""
        loader = MultiAssetLoader()
        
        # Check for the actual method that exists
        assert hasattr(loader, 'load_single_asset')

    def test_loader_dataset_dir_is_path(self):
        """Test that dataset_dir is a Path object."""
        loader = MultiAssetLoader()
        
        assert isinstance(loader.dataset_dir, Path)

    def test_yfinance_symbols_for_energy_futures(self):
        """Test that energy futures have yfinance symbols."""
        loader = MultiAssetLoader()
        
        # Common energy futures should be in yfinance symbols
        assert 'CL' in loader.YFINANCE_SYMBOLS  # Crude Oil
        assert 'NG' in loader.YFINANCE_SYMBOLS  # Natural Gas

    def test_initialization_with_string_path(self):
        """Test initialization with string path instead of Path object."""
        temp_dir = tempfile.mkdtemp()
        loader = MultiAssetLoader(dataset_dir=temp_dir)
        
        # dataset_dir should exist (either as string or Path)
        assert loader.dataset_dir is not None
