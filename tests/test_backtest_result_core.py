"""
Comprehensive unit tests for BacktestResult.
Tests for backtest result analysis and metric calculation.
"""

import pytest
import pandas as pd
import numpy as np
from core.portfolio.backtest_result import BacktestResult


class TestBacktestResultProperties:
    """Test suite for BacktestResult properties."""

    def test_final_equity_with_data(self):
        """
        Test final_equity property with valid equity curve.
        Should return last value in TotalValue column.
        """
        # Arrange
        equity = pd.DataFrame({
            'TotalValue': [100000, 105000, 110000]
        }, index=pd.date_range('2023-01-01', periods=3))
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act & Assert
        assert result.final_equity == 110000

    def test_final_equity_empty_curve(self):
        """
        Test final_equity with empty equity curve.
        Should return initial capital.
        """
        # Arrange
        result = BacktestResult(
            equity_curve=pd.DataFrame(),
            trades=pd.DataFrame(),
            initial_capital=100000
        )
        
        # Act & Assert
        assert result.final_equity == 100000

    def test_total_return_positive(self):
        """
        Test total_return with profitable backtest.
        Should return positive decimal (e.g., 0.10 for 10% gain).
        """
        # Arrange
        equity = pd.DataFrame({
            'TotalValue': [100000, 105000, 110000]
        }, index=pd.date_range('2023-01-01', periods=3))
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        total_ret = result.total_return
        
        # Assert
        assert pytest.approx(total_ret, rel=1e-4) == 0.10

    def test_total_return_negative(self):
        """
        Test total_return with losing backtest.
        Should return negative decimal.
        """
        # Arrange
        equity = pd.DataFrame({
            'TotalValue': [100000, 95000, 90000]
        }, index=pd.date_range('2023-01-01', periods=3))
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        total_ret = result.total_return
        
        # Assert
        assert pytest.approx(total_ret, rel=1e-4) == -0.10

    def test_total_return_empty_curve(self):
        """
        Test total_return with empty equity curve.
        Should return 0.0.
        """
        # Arrange
        result = BacktestResult(equity_curve=pd.DataFrame(), trades=pd.DataFrame())
        
        # Act & Assert
        assert result.total_return == 0.0

    def test_returns_series(self):
        """
        Test returns property generates daily return series.
        Should return pct_change of equity curve.
        """
        # Arrange
        equity = pd.DataFrame({
            'TotalValue': [100000, 101000, 102010, 101000]
        }, index=pd.date_range('2023-01-01', periods=4))
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        returns = result.returns
        
        # Assert
        assert len(returns) == 3  # dropna removes first NaN
        assert pytest.approx(returns.iloc[0], rel=1e-4) == 0.01
        assert pytest.approx(returns.iloc[1], rel=1e-4) == 0.01

    def test_returns_empty_curve(self):
        """
        Test returns with empty equity curve.
        Should return empty Series.
        """
        # Arrange
        result = BacktestResult(equity_curve=pd.DataFrame(), trades=pd.DataFrame())
        
        # Act
        returns = result.returns
        
        # Assert
        assert len(returns) == 0


class TestBacktestResultMetrics:
    """Test suite for metric calculations."""

    def test_metrics_empty_curve(self):
        """
        Test metrics with empty equity curve.
        Should return dict with all zeros.
        """
        # Arrange
        result = BacktestResult(equity_curve=pd.DataFrame(), trades=pd.DataFrame())
        
        # Act
        metrics = result.metrics
        
        # Assert
        assert metrics['Total Return'] == 0.0
        assert metrics['CAGR'] == 0.0
        assert metrics['Sharpe Ratio'] == 0.0
        assert metrics['Max Drawdown'] == 0.0
        assert metrics['Total Trades'] == 0

    def test_sharpe_ratio_calculation(self):
        """
        Test Sharpe ratio calculation.
        Should use 2% risk-free rate and annualize properly.
        """
        # Arrange
        # Create 252 days of 0.5% daily returns (very high)
        returns = np.full(252, 0.005)
        equity_values = 100000 * (1 + returns).cumprod()
        equity_values = np.insert(equity_values, 0, 100000)  # Add initial value
        
        equity = pd.DataFrame({
            'TotalValue': equity_values
        }, index=pd.date_range('2023-01-01', periods=253))
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        sharpe = result.metrics['Sharpe Ratio']
        
        # Assert
        # With constant 0.5% daily returns and low volatility, Sharpe should be very high
        assert sharpe > 5.0

    def test_sharpe_ratio_zero_returns(self):
        """
        Test Sharpe ratio with zero returns.
        Should handle edge case without division errors.
        """
        # Arrange
        equity = pd.DataFrame({
            'TotalValue': [100000] * 10  # Flat equity
        }, index=pd.date_range('2023-01-01', periods=10))
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        sharpe = result.metrics['Sharpe Ratio']
        
        # Assert
        assert sharpe == 0.0

    def test_sortino_ratio_calculation(self):
        """
        Test Sortino ratio (only penalizes downside volatility).
        Should be calculable for mixed returns.
        """
        # Arrange
        # Mix of positive and negative returns with more upside
        returns = [0.01, 0.02, -0.005, 0.015, -0.003, 0.02] * 50
        equity_values = 100000 * (1 + pd.Series(returns)).cumprod()
        equity_values = pd.concat([pd.Series([100000]), equity_values]).reset_index(drop=True)
        
        equity = pd.DataFrame({
            'TotalValue': equity_values
        }, index=pd.date_range('2023-01-01', periods=len(equity_values)))
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        sortino = result.metrics['Sortino Ratio']
        
        # Assert
        # With positive expected returns and some downside, sortino should be positive
        assert isinstance(sortino, (int, float))

    def test_sortino_ratio_no_downside(self):
        """
        Test Sortino ratio when no negative returns.
        Should handle edge case gracefully.
        """
        # Arrange
        equity = pd.DataFrame({
            'TotalValue': [100000, 101000, 102000, 103000]  # Only positive returns
        }, index=pd.date_range('2023-01-01', periods=4))
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        sortino = result.metrics['Sortino Ratio']
        
        # Assert
        assert sortino == 0.0  # No downside volatility

    def test_max_drawdown_calculation(self):
        """
        Test maximum drawdown calculation.
        Should identify worst peak-to-trough decline.
        """
        # Arrange
        equity = pd.DataFrame({
            'TotalValue': [100000, 110000, 105000, 95000, 100000, 105000]
        }, index=pd.date_range('2023-01-01', periods=6))
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        max_dd = result.metrics['Max Drawdown']
        
        # Assert
        # Drawdown from 110k to 95k = -13.6%
        assert pytest.approx(max_dd, rel=1e-2) == -0.136

    def test_max_drawdown_no_drawdown(self):
        """
        Test max drawdown with only increasing equity.
        Should return 0.0 or small negative number.
        """
        # Arrange
        equity = pd.DataFrame({
            'TotalValue': [100000, 105000, 110000, 115000]
        }, index=pd.date_range('2023-01-01', periods=4))
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        max_dd = result.metrics['Max Drawdown']
        
        # Assert
        assert max_dd <= 0.0

    def test_cagr_calculation(self):
        """
        Test CAGR calculation.
        Should compound annual growth rate over period.
        """
        # Arrange
        # Create proper equity curve with 252 trading days = 1 year
        dates = pd.date_range('2023-01-01', periods=253, freq='D')
        equity_values = np.linspace(100000, 110000, 253)  # Linear growth to 10%
        
        equity = pd.DataFrame({
            'TotalValue': equity_values
        }, index=dates)
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        cagr = result.metrics['CAGR']
        
        # Assert
        # Should be approximately 10% for 1 year
        assert pytest.approx(cagr, rel=1e-1) == 0.10

    def test_cagr_multiple_years(self):
        """
        Test CAGR over multiple years.
        Should annualize properly.
        """
        # Arrange
        # 504 days (2 years), 21% total return → ~10% CAGR
        dates = pd.date_range('2023-01-01', periods=505, freq='D')
        equity_values = np.linspace(100000, 121000, 505)
        
        equity = pd.DataFrame({
            'TotalValue': equity_values
        }, index=dates)
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        cagr = result.metrics['CAGR']
        
        # Assert
        assert pytest.approx(cagr, rel=1e-1) == 0.10

    def test_calmar_ratio_calculation(self):
        """
        Test Calmar ratio (CAGR / Max Drawdown).
        Should be ratio of return to risk.
        """
        # Arrange
        # Create equity curve with known CAGR and drawdown
        equity = pd.DataFrame({
            'TotalValue': [100000, 110000, 100000, 120000]  # 20% total, but had drawdown
        }, index=pd.date_range('2023-01-01', periods=4, freq='90D'))
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        calmar = result.metrics['Calmar Ratio']
        cagr = result.metrics['CAGR']
        max_dd = abs(result.metrics['Max Drawdown'])
        
        # Assert
        if max_dd > 0:
            assert pytest.approx(calmar, rel=1e-2) == cagr / max_dd

    def test_calmar_ratio_no_drawdown(self):
        """
        Test Calmar ratio when no drawdown.
        Should return 0.0 (division by zero case).
        """
        # Arrange
        equity = pd.DataFrame({
            'TotalValue': [100000, 105000, 110000]
        }, index=pd.date_range('2023-01-01', periods=3))
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        calmar = result.metrics['Calmar Ratio']
        
        # Assert
        assert calmar == 0.0


class TestBacktestResultTrades:
    """Test suite for trade-based metrics."""

    def test_win_rate_all_winners(self):
        """
        Test win rate with 100% winning trades.
        Should return 1.0.
        """
        # Arrange
        trades = pd.DataFrame({
            'pnl': [100, 200, 150, 75]
        })
        result = BacktestResult(
            equity_curve=pd.DataFrame({'TotalValue': [100000]}),
            trades=trades
        )
        
        # Act
        win_rate = result.metrics['Win Rate']
        
        # Assert
        assert win_rate == 1.0

    def test_win_rate_all_losers(self):
        """
        Test win rate with 100% losing trades.
        Should return 0.0.
        """
        # Arrange
        trades = pd.DataFrame({
            'pnl': [-100, -200, -150, -75]
        })
        result = BacktestResult(
            equity_curve=pd.DataFrame({'TotalValue': [100000]}),
            trades=trades
        )
        
        # Act
        win_rate = result.metrics['Win Rate']
        
        # Assert
        assert win_rate == 0.0

    def test_win_rate_mixed(self):
        """
        Test win rate with mixed trades.
        Should return correct percentage.
        """
        # Arrange
        trades = pd.DataFrame({
            'pnl': [100, -50, 200, -75, 150]  # 3 winners, 2 losers
        })
        result = BacktestResult(
            equity_curve=pd.DataFrame({'TotalValue': [100000]}),
            trades=trades
        )
        
        # Act
        win_rate = result.metrics['Win Rate']
        
        # Assert
        assert pytest.approx(win_rate, rel=1e-4) == 0.6

    def test_win_rate_no_trades(self):
        """
        Test win rate with no trades.
        Should return 0.0.
        """
        # Arrange
        result = BacktestResult(
            equity_curve=pd.DataFrame({'TotalValue': [100000]}),
            trades=pd.DataFrame()
        )
        
        # Act
        win_rate = result.metrics['Win Rate']
        
        # Assert
        assert win_rate == 0.0

    def test_avg_trade_calculation(self):
        """
        Test average trade P&L.
        Should return mean of all trades.
        """
        # Arrange
        trades = pd.DataFrame({
            'pnl': [100, -50, 200, -100, 50]  # Avg = 40
        })
        result = BacktestResult(
            equity_curve=pd.DataFrame({'TotalValue': [100000]}),
            trades=trades
        )
        
        # Act
        avg_trade = result.metrics['Avg Trade']
        
        # Assert
        assert pytest.approx(avg_trade, rel=1e-4) == 40.0

    def test_profit_factor_calculation(self):
        """
        Test profit factor (gross profits / gross losses).
        Should be ratio of winning to losing trade totals.
        """
        # Arrange
        trades = pd.DataFrame({
            'pnl': [100, 200, -50, -75]  # Wins: 300, Losses: 125, PF: 2.4
        })
        result = BacktestResult(
            equity_curve=pd.DataFrame({'TotalValue': [100000]}),
            trades=trades
        )
        
        # Act
        pf = result.metrics['Profit Factor']
        
        # Assert
        assert pytest.approx(pf, rel=1e-4) == 2.4

    def test_profit_factor_no_losses(self):
        """
        Test profit factor when no losing trades.
        Should return inf.
        """
        # Arrange
        trades = pd.DataFrame({
            'pnl': [100, 200, 150]
        })
        result = BacktestResult(
            equity_curve=pd.DataFrame({'TotalValue': [100000]}),
            trades=trades
        )
        
        # Act
        pf = result.metrics['Profit Factor']
        
        # Assert
        assert pf == float('inf')

    def test_profit_factor_no_wins(self):
        """
        Test profit factor when no winning trades.
        Should return 0.0.
        """
        # Arrange
        trades = pd.DataFrame({
            'pnl': [-100, -200, -150]
        })
        result = BacktestResult(
            equity_curve=pd.DataFrame({'TotalValue': [100000]}),
            trades=trades
        )
        
        # Act
        pf = result.metrics['Profit Factor']
        
        # Assert
        assert pf == 0.0

    def test_profit_factor_no_trades(self):
        """
        Test profit factor with no trades.
        Should return 0.0.
        """
        # Arrange
        result = BacktestResult(
            equity_curve=pd.DataFrame({'TotalValue': [100000]}),
            trades=pd.DataFrame()
        )
        
        # Act
        pf = result.metrics['Profit Factor']
        
        # Assert
        assert pf == 0.0


class TestBacktestResultEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_day_equity_curve(self):
        """
        Test with single day (no returns to calculate).
        Should handle gracefully without errors.
        """
        # Arrange
        equity = pd.DataFrame({
            'TotalValue': [100000]
        }, index=[pd.Timestamp('2023-01-01')])
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        metrics = result.metrics
        
        # Assert
        assert metrics['Total Return'] == 0.0
        assert len(result.returns) == 0

    def test_zero_initial_value(self):
        """
        Test edge case with zero starting value.
        Should avoid division by zero errors.
        """
        # Arrange
        equity = pd.DataFrame({
            'TotalValue': [0, 100, 200]
        }, index=pd.date_range('2023-01-01', periods=3))
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        total_ret = result.total_return
        cagr = result.metrics['CAGR']
        
        # Assert
        # Should handle gracefully, likely returning 0 or special value
        assert isinstance(total_ret, (int, float))
        assert isinstance(cagr, (int, float))

    def test_very_short_period(self):
        """
        Test CAGR calculation with very short period.
        Should handle periods < 1 year properly.
        """
        # Arrange
        equity = pd.DataFrame({
            'TotalValue': [100000, 110000]  # 10% gain over 10 days
        }, index=[pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-11')])
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        # Act
        cagr = result.metrics['CAGR']
        
        # Assert
        # CAGR should be annualized, so much higher than 10%
        assert cagr > 0.10
