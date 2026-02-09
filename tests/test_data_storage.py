"""
Tests for core.data.storage.parquet_handler module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from core.data.storage.parquet_handler import ParquetHandler


class TestParquetHandler:
    """Test suite for ParquetHandler class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        # Cleanup after tests
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def handler(self, temp_dir):
        """Create ParquetHandler instance with temporary directory."""
        return ParquetHandler(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        return pd.DataFrame({
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)

    def test_initialization_creates_directory(self, temp_dir):
        """Test that ParquetHandler creates the base directory on initialization."""
        test_path = Path(temp_dir) / "new_directory"
        handler = ParquetHandler(test_path)
        
        assert handler.base_path == test_path
        assert test_path.exists()
        assert test_path.is_dir()

    def test_save_and_load_basic(self, handler, sample_data):
        """Test basic save and load functionality."""
        filename = "test_data.parquet"
        
        # Save data
        handler.save(sample_data, filename)
        
        # Verify file exists
        assert (handler.base_path / filename).exists()
        
        # Load data
        loaded_data = handler.load(filename)
        
        # Verify loaded data matches original (check_freq=False for parquet compatibility)
        pd.testing.assert_frame_equal(loaded_data, sample_data, check_freq=False)

    def test_load_nonexistent_file_returns_empty_dataframe(self, handler):
        """Test that loading a non-existent file returns an empty DataFrame."""
        result = handler.load("nonexistent.parquet")
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_save_with_path_object(self, temp_dir, sample_data):
        """Test that handler accepts Path objects."""
        handler = ParquetHandler(Path(temp_dir))
        filename = "test_path.parquet"
        
        handler.save(sample_data, filename)
        loaded_data = handler.load(filename)
        
        pd.testing.assert_frame_equal(loaded_data, sample_data, check_freq=False)

    def test_save_empty_dataframe(self, handler):
        """Test saving an empty DataFrame."""
        empty_df = pd.DataFrame()
        filename = "empty.parquet"
        
        handler.save(empty_df, filename)
        loaded_data = handler.load(filename)
        
        assert loaded_data.empty

    def test_save_with_different_dtypes(self, handler):
        """Test saving DataFrame with mixed data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        filename = "mixed_types.parquet"
        
        handler.save(df, filename)
        loaded_data = handler.load(filename)
        
        pd.testing.assert_frame_equal(loaded_data, df)

    def test_save_with_multiindex(self, handler):
        """Test saving DataFrame with MultiIndex."""
        idx = pd.MultiIndex.from_product([['A', 'B'], [1, 2]], names=['letter', 'number'])
        df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=idx)
        filename = "multiindex.parquet"
        
        handler.save(df, filename)
        loaded_data = handler.load(filename)
        
        pd.testing.assert_frame_equal(loaded_data, df)

    def test_overwrite_existing_file(self, handler, sample_data):
        """Test that saving to an existing filename overwrites the file."""
        filename = "overwrite_test.parquet"
        
        # Save original data
        handler.save(sample_data, filename)
        
        # Create different data
        new_data = pd.DataFrame({'value': [999]})
        
        # Overwrite
        handler.save(new_data, filename)
        loaded_data = handler.load(filename)
        
        pd.testing.assert_frame_equal(loaded_data, new_data)

    def test_handler_with_nested_path(self, temp_dir, sample_data):
        """Test ParquetHandler with nested directory structure."""
        nested_path = Path(temp_dir) / "level1" / "level2"
        handler = ParquetHandler(nested_path)
        filename = "nested.parquet"
        
        handler.save(sample_data, filename)
        
        assert nested_path.exists()
        assert (nested_path / filename).exists()
        
        loaded_data = handler.load(filename)
        pd.testing.assert_frame_equal(loaded_data, sample_data, check_freq=False)

    def test_save_invalid_data_raises_error(self, handler):
        """Test that saving invalid data raises an error."""
        filename = "invalid.parquet"
        
        with pytest.raises(Exception):
            handler.save("not a dataframe", filename)

    def test_load_corrupted_file_raises_error(self, handler):
        """Test that loading a corrupted file raises an error."""
        filename = "corrupted.parquet"
        
        # Create a corrupted file
        (handler.base_path / filename).write_text("this is not a valid parquet file")
        
        with pytest.raises(Exception):
            handler.load(filename)
