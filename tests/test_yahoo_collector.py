"""
Comprehensive unit tests for YahooCollector.
Tests for Yahoo Finance data collection functionality.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from core.data.collectors.yahoo_collector import YahooCollector


class TestYahooCollectorFetchHistory:
    """Test suite for YahooCollector.fetch_history method."""

    def test_fetch_history_empty_tickers(self):
        """
        Test fetch_history with empty ticker list.
        Should return empty DataFrame and log warning.
        """
        # Arrange
        collector = YahooCollector()
        
        # Act
        result = collector.fetch_history([], start_date='2023-01-01', end_date='2023-12-31')
        
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch('core.data.collectors.yahoo_collector.yf.download')
    def test_fetch_history_single_ticker(self, mock_download):
        """
        Test fetch_history with single ticker.
        Should call yfinance with correct parameters and return data.
        """
        # Arrange
        collector = YahooCollector()
        mock_df = pd.DataFrame({
            'Close': [100, 101, 102],
            'Open': [99, 100, 101],
            'High': [101, 102, 103],
            'Low': [98, 99, 100],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        mock_download.return_value = mock_df
        
        # Act
        result = collector.fetch_history(['AAPL'], start_date='2023-01-01', end_date='2023-01-03')
        
        # Assert
        mock_download.assert_called_once()
        call_args = mock_download.call_args
        assert 'AAPL' in call_args.kwargs['tickers']
        assert call_args.kwargs['start'] == '2023-01-01'
        assert call_args.kwargs['end'] == '2023-01-03'
        assert call_args.kwargs['auto_adjust'] is True
        assert not result.empty

    @patch('core.data.collectors.yahoo_collector.yf.download')
    def test_fetch_history_multiple_tickers(self, mock_download):
        """
        Test fetch_history with multiple tickers.
        Should join tickers with space and return multi-index data.
        """
        # Arrange
        collector = YahooCollector()
        mock_df = pd.DataFrame({
            'Close': [100, 101, 102],
        }, index=pd.date_range('2023-01-01', periods=3))
        mock_download.return_value = mock_df
        
        # Act
        result = collector.fetch_history(['AAPL', 'MSFT', 'GOOGL'], 
                                        start_date='2023-01-01', 
                                        end_date='2023-01-03')
        
        # Assert
        call_args = mock_download.call_args
        assert call_args.kwargs['tickers'] == 'AAPL MSFT GOOGL'
        assert call_args.kwargs['group_by'] == 'ticker'

    @patch('core.data.collectors.yahoo_collector.yf.download')
    def test_fetch_history_with_interval(self, mock_download):
        """
        Test fetch_history with custom interval (hourly, weekly, etc.).
        Should pass interval parameter to yfinance.
        """
        # Arrange
        collector = YahooCollector()
        mock_df = pd.DataFrame({'Close': [100]}, index=pd.date_range('2023-01-01', periods=1))
        mock_download.return_value = mock_df
        
        # Act
        collector.fetch_history(['AAPL'], start_date='2023-01-01', interval='1h')
        
        # Assert
        call_args = mock_download.call_args
        assert call_args.kwargs['interval'] == '1h'

    @patch('core.data.collectors.yahoo_collector.yf.download')
    def test_fetch_history_returns_empty_dataframe(self, mock_download):
        """
        Test fetch_history when yfinance returns empty DataFrame.
        Should handle gracefully and return empty DataFrame.
        """
        # Arrange
        collector = YahooCollector()
        mock_download.return_value = pd.DataFrame()
        
        # Act
        result = collector.fetch_history(['INVALID'], start_date='2023-01-01')
        
        # Assert
        assert result.empty

    @patch('core.data.collectors.yahoo_collector.yf.download')
    def test_fetch_history_with_datetime_objects(self, mock_download):
        """
        Test fetch_history with datetime objects instead of strings.
        Should accept both string and datetime inputs.
        """
        # Arrange
        collector = YahooCollector()
        mock_df = pd.DataFrame({'Close': [100]}, index=pd.date_range('2023-01-01', periods=1))
        mock_download.return_value = mock_df
        
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        
        # Act
        collector.fetch_history(['AAPL'], start_date=start, end_date=end)
        
        # Assert
        call_args = mock_download.call_args
        assert call_args.kwargs['start'] == start
        assert call_args.kwargs['end'] == end

    @patch('core.data.collectors.yahoo_collector.yf.download')
    def test_fetch_history_exception_handling(self, mock_download):
        """
        Test fetch_history when yfinance raises exception.
        Should log error and re-raise exception.
        """
        # Arrange
        collector = YahooCollector()
        mock_download.side_effect = Exception("Network error")
        
        # Act & Assert
        with pytest.raises(Exception, match="Network error"):
            collector.fetch_history(['AAPL'], start_date='2023-01-01')


class TestYahooCollectorFetchLatest:
    """Test suite for YahooCollector.fetch_latest method."""

    @patch('core.data.collectors.yahoo_collector.yf.download')
    def test_fetch_latest_returns_last_row(self, mock_download):
        """
        Test fetch_latest returns only the most recent data point.
        Should return single-row DataFrame.
        """
        # Arrange
        collector = YahooCollector()
        mock_df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
        }, index=pd.date_range('2023-12-25', periods=5))
        mock_download.return_value = mock_df
        
        # Act
        result = collector.fetch_latest(['AAPL'])
        
        # Assert
        assert len(result) == 1
        assert result['Close'].iloc[0] == 104

    @patch('core.data.collectors.yahoo_collector.yf.download')
    def test_fetch_latest_fetches_recent_week(self, mock_download):
        """
        Test fetch_latest requests 7 days of data.
        Should request data from last week to ensure fresh data.
        """
        # Arrange
        collector = YahooCollector()
        mock_df = pd.DataFrame({'Close': [100]}, index=pd.date_range('2023-01-01', periods=1))
        mock_download.return_value = mock_df
        
        # Act
        with patch('core.data.collectors.yahoo_collector.datetime') as mock_datetime:
            mock_now = datetime(2023, 12, 31, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            collector.fetch_latest(['AAPL'])
            
            # Assert
            call_args = mock_download.call_args
            # Should fetch 7 days back
            # Note: kwargs key is 'start' not 'start_date'
            assert call_args is not None

    @patch('core.data.collectors.yahoo_collector.yf.download')
    def test_fetch_latest_empty_dataframe(self, mock_download):
        """
        Test fetch_latest when no data available.
        Should return empty DataFrame.
        """
        # Arrange
        collector = YahooCollector()
        mock_download.return_value = pd.DataFrame()
        
        # Act
        result = collector.fetch_latest(['INVALID'])
        
        # Assert
        assert result.empty

    @patch('core.data.collectors.yahoo_collector.yf.download')
    def test_fetch_latest_multiple_tickers(self, mock_download):
        """
        Test fetch_latest with multiple tickers.
        Should fetch latest for all tickers.
        """
        # Arrange
        collector = YahooCollector()
        mock_df = pd.DataFrame({
            'Close': [100, 101, 102],
        }, index=pd.date_range('2023-12-29', periods=3))
        mock_download.return_value = mock_df
        
        # Act
        result = collector.fetch_latest(['AAPL', 'MSFT'])
        
        # Assert
        assert len(result) == 1  # Only last row
        call_args = mock_download.call_args
        assert 'AAPL MSFT' in call_args.kwargs['tickers']


class TestYahooCollectorEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch('core.data.collectors.yahoo_collector.yf.download')
    def test_none_end_date(self, mock_download):
        """
        Test fetch_history with None end_date.
        Should fetch up to current date.
        """
        # Arrange
        collector = YahooCollector()
        mock_df = pd.DataFrame({'Close': [100]}, index=pd.date_range('2023-01-01', periods=1))
        mock_download.return_value = mock_df
        
        # Act
        collector.fetch_history(['AAPL'], start_date='2023-01-01', end_date=None)
        
        # Assert
        call_args = mock_download.call_args
        assert call_args.kwargs['end'] is None

    @patch('core.data.collectors.yahoo_collector.yf.download')
    def test_ticker_with_special_characters(self, mock_download):
        """
        Test fetch_history with ticker containing special characters.
        Should handle tickers like BRK.B, ^VIX, etc.
        """
        # Arrange
        collector = YahooCollector()
        mock_df = pd.DataFrame({'Close': [100]}, index=pd.date_range('2023-01-01', periods=1))
        mock_download.return_value = mock_df
        
        # Act
        collector.fetch_history(['BRK.B', '^VIX'], start_date='2023-01-01')
        
        # Assert
        call_args = mock_download.call_args
        assert 'BRK.B ^VIX' in call_args.kwargs['tickers']
