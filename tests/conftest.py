"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides:
- Custom markers for test categorization
- Shared fixtures available to all tests
- Test environment configuration
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests across multiple components"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take >1 second to run"
    )


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def mock_ohlcv_data():
    """Generate realistic mock OHLCV data for testing.
    
    Returns 1 year of daily data with realistic price movements.
    """
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    np.random.seed(42)
    base_price = 100.0
    daily_returns = np.random.normal(0.0005, 0.015, len(dates))
    prices = base_price * (1 + daily_returns).cumprod()
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        'High': prices * (1 + np.random.uniform(0, 0.015, len(dates))),
        'Low': prices * (1 - np.random.uniform(0, 0.015, len(dates))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Ensure OHLC relationships are valid
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df.set_index('Date')


@pytest.fixture
def mock_price_history():
    """Generate price history for multiple assets.
    
    Returns dict of DataFrames with OHLCV data for AAPL, MSFT, GOOGL.
    """
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    prices = {}
    np.random.seed(42)
    
    for i, ticker in enumerate(tickers):
        base_price = 100.0 * (i + 1)  # Different price levels
        drift = 0.0005 + i * 0.0002  # Different trends
        vol = 0.015 + i * 0.005  # Different volatilities
        
        returns = np.random.normal(drift, vol, len(dates))
        close_prices = base_price * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'Open': close_prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
            'High': close_prices * (1 + np.random.uniform(0, 0.015, len(dates))),
            'Low': close_prices * (1 - np.random.uniform(0, 0.015, len(dates))),
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        
        prices[ticker] = df
    
    return prices


@pytest.fixture
def mock_signals():
    """Generate mock trading signals for testing.
    
    Returns DataFrame with Date index and Signal column [-1, 1].
    """
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Oscillating signals with some noise
    np.random.seed(42)
    signals = np.sin(np.arange(len(dates)) * 0.1) + np.random.normal(0, 0.2, len(dates))
    signals = np.clip(signals, -1, 1)
    
    return pd.DataFrame({'Signal': signals}, index=dates)


# ============================================================================
# Comparison Utilities
# ============================================================================

@pytest.fixture
def approx():
    """Wrapper for pytest.approx with standard tolerances."""
    def _approx(value, rel=1e-4, abs=1e-6):
        return pytest.approx(value, rel=rel, abs=abs)
    return _approx


@pytest.fixture
def mock_matplotlib_axes():
    """Fixture providing properly configured matplotlib mocks.
    
    Returns:
        dict: Dictionary containing:
            - 'fig': Mock figure object
            - 'axes': NumPy array of [mock_ax1, mock_ax2]
            - 'ax1': First mock axes
            - 'ax2': Second mock axes
    
    Example:
        def test_plotting(self, mock_matplotlib_axes):
            mocks = mock_matplotlib_axes
            with patch('matplotlib.pyplot.subplots', return_value=(mocks['fig'], mocks['axes'])):
                # Use in test
                pass
    """
    from unittest.mock import MagicMock
    
    mock_fig = MagicMock()
    mock_ax1 = MagicMock()
    mock_ax2 = MagicMock()
    
    # Configure all common matplotlib methods
    for ax in [mock_ax1, mock_ax2]:
        ax.figure = mock_fig
        ax.get_figure = MagicMock(return_value=mock_fig)
        ax.xaxis = MagicMock()
        ax.yaxis = MagicMock()
        ax.yaxis.set_major_formatter = MagicMock()
        ax.axhline = MagicMock(return_value=None)
        ax.axvline = MagicMock(return_value=None)
        ax.plot = MagicMock(return_value=[MagicMock()])
        ax.set_title = MagicMock(return_value=None)
        ax.set_xlabel = MagicMock(return_value=None)
        ax.set_ylabel = MagicMock(return_value=None)
        ax.legend = MagicMock(return_value=None)
        ax.grid = MagicMock(return_value=None)
        ax.fill_between = MagicMock(return_value=None)
    
    axes = np.array([mock_ax1, mock_ax2])
    # Critical: Add figure attribute to the array itself
    axes.figure = mock_fig
    
    return {
        'fig': mock_fig,
        'axes': axes,
        'ax1': mock_ax1,
        'ax2': mock_ax2
    }
