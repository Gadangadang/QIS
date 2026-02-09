"""
Tests for core.data.processors.price_processor module.
"""

import pytest
import pandas as pd
import numpy as np

from core.data.processors.price_processor import PriceProcessor


class TestPriceProcessor:
    """Test suite for PriceProcessor class."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'High': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
            'Low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'Close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'Volume': [1000000] * 10
        }, index=dates)
        return df

    @pytest.fixture
    def data_with_missing_values(self):
        """Create data with missing values."""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        df = pd.DataFrame({
            'Close': [100.0, 101.0, np.nan, 103.0, np.nan, 105.0, 106.0, 107.0, 108.0, 109.0],
            'Volume': [1000000] * 10
        }, index=dates)
        return df

    @pytest.fixture
    def unsorted_data(self):
        """Create unsorted data."""
        dates = pd.date_range(start='2023-01-01', end='2023-01-05', freq='D')
        df = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0, 103.0, 104.0]
        }, index=dates)
        # Shuffle the index
        return df.sample(frac=1)

    def test_initialization_default_fill_method(self):
        """Test PriceProcessor initialization with default fill method."""
        processor = PriceProcessor()
        assert processor.fill_method == 'ffill'

    def test_initialization_custom_fill_method(self):
        """Test PriceProcessor initialization with custom fill method."""
        processor = PriceProcessor(fill_method='bfill')
        assert processor.fill_method == 'bfill'

    def test_process_empty_dataframe(self):
        """Test processing an empty DataFrame returns empty DataFrame."""
        processor = PriceProcessor()
        empty_df = pd.DataFrame()
        result = processor.process(empty_df)
        
        assert result.empty

    def test_process_fills_missing_values_ffill(self, data_with_missing_values):
        """Test that process fills missing values using forward fill."""
        processor = PriceProcessor(fill_method='ffill')
        result = processor.process(data_with_missing_values)
        
        # Check no missing values after processing
        assert not result['Close'].isna().any()
        
        # Check forward fill worked correctly
        assert result['Close'].iloc[2] == 101.0  # Filled with previous value
        assert result['Close'].iloc[4] == 103.0  # Filled with previous value

    def test_process_converts_to_datetime_index(self):
        """Test that process converts index to DatetimeIndex."""
        df = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0]
        }, index=['2023-01-01', '2023-01-02', '2023-01-03'])
        
        processor = PriceProcessor()
        result = processor.process(df)
        
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_process_sorts_index(self, unsorted_data):
        """Test that process sorts the index."""
        processor = PriceProcessor()
        result = processor.process(unsorted_data)
        
        # Check that index is sorted
        assert result.index.is_monotonic_increasing
        
        # Check that first date is earliest
        assert result.index[0] == pd.Timestamp('2023-01-01')

    def test_process_preserves_data(self, sample_ohlcv_data):
        """Test that process preserves all columns and data."""
        processor = PriceProcessor()
        result = processor.process(sample_ohlcv_data)
        
        # Check all columns are preserved
        assert list(result.columns) == list(sample_ohlcv_data.columns)
        
        # Check data is preserved (sorted)
        pd.testing.assert_frame_equal(result, sample_ohlcv_data)

    def test_calculate_returns_default_period(self, sample_ohlcv_data):
        """Test calculate_returns with default period (1)."""
        processor = PriceProcessor()
        result = processor.calculate_returns(sample_ohlcv_data)
        
        # First row should be NaN
        assert result.iloc[0].isna().all()
        
        # Check calculation for second row
        expected_return = (102.0 - 101.0) / 101.0
        assert abs(result['Close'].iloc[1] - expected_return) < 1e-6

    def test_calculate_returns_custom_period(self, sample_ohlcv_data):
        """Test calculate_returns with custom period."""
        processor = PriceProcessor()
        result = processor.calculate_returns(sample_ohlcv_data, period=2)
        
        # First two rows should be NaN
        assert result.iloc[0].isna().all()
        assert result.iloc[1].isna().all()
        
        # Check calculation for third row (2-day return)
        expected_return = (103.0 - 101.0) / 101.0
        assert abs(result['Close'].iloc[2] - expected_return) < 1e-6

    def test_resample_weekly(self, sample_ohlcv_data):
        """Test resampling to weekly frequency."""
        processor = PriceProcessor()
        result = processor.resample(sample_ohlcv_data, rule='W-FRI')
        
        # Should have fewer rows than daily data
        assert len(result) <= len(sample_ohlcv_data)
        
        # Check result is not empty
        assert not result.empty

    def test_resample_monthly(self, sample_ohlcv_data):
        """Test resampling to monthly frequency."""
        processor = PriceProcessor()
        result = processor.resample(sample_ohlcv_data, rule='M')  # M works in older pandas
        
        # Should have fewer rows
        assert len(result) <= len(sample_ohlcv_data)

    def test_process_does_not_modify_input(self, sample_ohlcv_data):
        """Test that process does not modify the input DataFrame."""
        processor = PriceProcessor()
        original = sample_ohlcv_data.copy()
        
        processor.process(sample_ohlcv_data)
        
        # Original should be unchanged
        pd.testing.assert_frame_equal(sample_ohlcv_data, original)

    def test_process_with_already_sorted_datetime_index(self, sample_ohlcv_data):
        """Test that process handles already-processed data correctly."""
        processor = PriceProcessor()
        
        # Process once
        result1 = processor.process(sample_ohlcv_data)
        
        # Process again (should be idempotent)
        result2 = processor.process(result1)
        
        pd.testing.assert_frame_equal(result1, result2)

    def test_calculate_returns_preserves_index(self, sample_ohlcv_data):
        """Test that calculate_returns preserves the index."""
        processor = PriceProcessor()
        result = processor.calculate_returns(sample_ohlcv_data)
        
        pd.testing.assert_index_equal(result.index, sample_ohlcv_data.index)

    def test_resample_preserves_column_names(self, sample_ohlcv_data):
        """Test that resample preserves column names."""
        processor = PriceProcessor()
        result = processor.resample(sample_ohlcv_data)
        
        # Column names should be preserved
        assert list(result.columns) == list(sample_ohlcv_data.columns)
