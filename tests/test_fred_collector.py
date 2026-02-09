"""
Comprehensive unit tests for FredCollector.
Tests for Federal Reserve Economic Data (FRED) collection functionality.
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch
from core.data.collectors.fred_collector import FredCollector


class TestFredCollectorFetchHistory:
    """Test suite for FredCollector.fetch_history method."""

    @patch('core.data.collectors.fred_collector.web.DataReader')
    def test_fetch_history_single_series(self, mock_get_data):
        """Test fetch_history with single FRED series."""
        collector = FredCollector()
        mock_df = pd.DataFrame({'DGS10': [2.5, 2.6, 2.7]}, 
                                index=pd.date_range('2023-01-01', periods=3))
        mock_get_data.return_value = mock_df
        
        result = collector.fetch_history(['DGS10'], start_date='2023-01-01', end_date='2023-01-03')
        
        assert not result.empty

    def test_fetch_history_empty_tickers(self):
        """Test fetch_history with empty series list."""
        collector = FredCollector()
        result = collector.fetch_history([], start_date='2023-01-01')
        assert result.empty

    @patch('core.data.collectors.fred_collector.web.DataReader')
    def test_fetch_history_exception_handling(self, mock_get_data):
        """Test exception handling."""
        collector = FredCollector()
        mock_get_data.side_effect = Exception("API error")
        
        with pytest.raises(Exception):
            collector.fetch_history(['DGS10'], start_date='2023-01-01')


class TestFredCollectorFetchLatest:
    """Test suite for FredCollector.fetch_latest method."""

    @patch('core.data.collectors.fred_collector.web.DataReader')
    def test_fetch_latest_returns_last_row(self, mock_get_data):
        """Test fetch_latest returns only most recent observation."""
        collector = FredCollector()
        mock_df = pd.DataFrame({'DGS10': [2.5, 2.6, 2.7, 2.8, 2.9]},
                                index=pd.date_range('2023-12-25', periods=5))
        mock_get_data.return_value = mock_df
        
        result = collector.fetch_latest(['DGS10'])
        
        assert len(result) == 1
        assert result['DGS10'].iloc[0] == 2.9

    @patch('core.data.collectors.fred_collector.web.DataReader')
    def test_fetch_latest_empty_dataframe(self, mock_get_data):
        """Test fetch_latest when no data available."""
        collector = FredCollector()
        mock_get_data.return_value = pd.DataFrame()
        
        result = collector.fetch_latest(['INVALID'])
        assert result.empty
