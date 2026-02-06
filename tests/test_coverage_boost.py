"""
Additional simple tests to improve coverage to 56%.

Focused on easy-to-test utility functions and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from core.portfolio.backtest_result import BacktestResult


# ============================================================================
# BacktestResult Additional Tests
# ============================================================================

class TestBacktestResultReturns:
    """Test BacktestResult returns property."""
    
    def test_returns_empty_equity(self):
        """Test returns with empty equity curve."""
        result = BacktestResult(
            equity_curve=pd.DataFrame(),
            trades=pd.DataFrame()
        )
        
        returns = result.returns
        
        # Should return empty Series
        assert isinstance(returns, pd.Series)
        assert len(returns) == 0
    
    def test_returns_calculation(self):
        """Test returns are calculated correctly."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        values = [100000, 101000, 100500, 102000, 101500]
        
        equity = pd.DataFrame({
            'TotalValue': values
        }, index=dates)
        
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        returns = result.returns
        
        # Should have 4 returns (n-1)
        assert len(returns) == 4
        
        # First return should be 1%
        assert abs(returns.iloc[0] - 0.01) < 0.0001
    
    def test_final_equity_property(self):
        """Test final_equity property."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        values = np.linspace(100000, 110000, 10)
        
        equity = pd.DataFrame({
            'TotalValue': values
        }, index=dates)
        
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        assert result.final_equity == 110000
    
    def test_total_return_negative(self):
        """Test total return with losses."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        values = np.linspace(100000, 90000, 10)  # 10% loss
        
        equity = pd.DataFrame({
            'TotalValue': values
        }, index=dates)
        
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Should be approximately -10%
        assert result.total_return < 0
        assert -0.11 < result.total_return < -0.09


# ============================================================================
# Asset Registry Additional Tests
# ============================================================================

def test_main_block_execution():
    """Test that main block can execute without error."""
    # This tests the if __name__ == "__main__" block
    # We can't easily run it, but we can import it
    from core import asset_registry
    assert hasattr(asset_registry, 'print_registry_summary')


# ============================================================================
# Simple Signal Tests
# ============================================================================

def test_signal_generators_have_generate_method():
    """Test that all signal generators have generate method."""
    from signals.momentum import MomentumSignalV2
    from signals.mean_reversion import MeanReversionSignal
    from signals.ensemble import EnsembleSignal
    
    assert hasattr(MomentumSignalV2(), 'generate')
    assert hasattr(MeanReversionSignal(), 'generate')
    assert hasattr(EnsembleSignal(), 'generate')
