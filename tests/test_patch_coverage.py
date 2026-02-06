"""
Tests targeting specific patch lines that lacked coverage.

Covers the deprecated-pandas-to-modern-pandas changes in:
- core/paper_trading_engine.py  (try/except import fallback)
- core/reporter.py              (.reindex().ffill() in multi-strategy report)
- core/portfolio/portfolio_manager_v2.py  (.reindex().ffill() benchmark alignment)
- core/risk_dashboard.py        (.ffill().fillna(0) in multi-strategy dashboard)
- utils/plotter.py              (.reindex().ffill() benchmark equity calculation)

Run with: pytest tests/test_patch_coverage.py -v
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

from core.portfolio.backtest_result import BacktestResult
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2
from core.reporter import Reporter
from core.risk_dashboard import RiskDashboard
from utils.plotter import PortfolioPlotter


# ============================================================================
# Shared fixtures
# ============================================================================

@pytest.fixture
def dates():
    return pd.date_range(start='2023-01-01', end='2023-06-30', freq='B')


@pytest.fixture
def make_backtest_result(dates):
    """Factory that builds a BacktestResult with a simple upward equity curve."""
    def _make(initial_capital=100_000, trend=0.0003, seed=42):
        np.random.seed(seed)
        returns = np.random.normal(trend, 0.01, len(dates))
        total_value = initial_capital * (1 + returns).cumprod()

        equity = pd.DataFrame({
            'Cash': initial_capital * 0.5,
            'PositionValue': total_value - initial_capital * 0.5,
            'TotalValue': total_value,
        }, index=dates)

        trades = pd.DataFrame({
            'entry_date': [dates[5], dates[50]],
            'exit_date': [dates[10], dates[55]],
            'ticker': ['TEST', 'TEST'],
            'action': ['BUY', 'SELL'],
            'shares': [10, -10],
            'entry_price': [100.0, 105.0],
            'exit_price': [105.0, 110.0],
            'price': [100.0, 105.0],
            'pnl': [50.0, 50.0],
            'return': [0.05, 0.048],
        })

        return BacktestResult(
            initial_capital=initial_capital,
            equity_curve=equity,
            trades=trades,
        )
    return _make


@pytest.fixture
def strategy_results(make_backtest_result):
    """Two-strategy dict in the format used across the codebase."""
    return {
        'Strategy_A': {
            'result': make_backtest_result(initial_capital=60_000, seed=1),
            'capital': 60_000,
        },
        'Strategy_B': {
            'result': make_backtest_result(initial_capital=40_000, seed=2),
            'capital': 40_000,
        },
    }


@pytest.fixture
def benchmark_equity(dates):
    """Benchmark equity DataFrame with a TotalValue column."""
    np.random.seed(99)
    values = 100_000 * (1 + np.random.normal(0.0002, 0.008, len(dates))).cumprod()
    return pd.DataFrame({'TotalValue': values}, index=dates)


# ============================================================================
# 1. core/paper_trading_engine.py — import fallback
# ============================================================================

class TestPaperTradingEngineImport:
    """Cover the try/except import block (lines 29-44)."""

    def test_module_imports_without_error(self):
        """The module should import cleanly even though archive modules are missing."""
        import core.paper_trading_engine as mod
        # The ImportError path sets these to None
        assert mod.PortfolioManager is None
        assert mod.PortfolioConfig is None
        assert mod.BacktestResult is None
        assert mod.RiskManager is None

    def test_paper_trading_state_roundtrip(self):
        """PaperTradingState can serialise and deserialise."""
        from core.paper_trading_engine import PaperTradingState

        state = PaperTradingState()
        state.cash = 50_000.0
        state.positions = {'AAPL': {'shares': 100}}
        state.initial_capital = 100_000.0

        d = state.to_dict()
        restored = PaperTradingState.from_dict(d)

        assert restored.cash == 50_000.0
        assert restored.positions == {'AAPL': {'shares': 100}}
        assert restored.initial_capital == 100_000.0

    def test_paper_trading_state_empty_roundtrip(self):
        """Default state round-trips cleanly."""
        from core.paper_trading_engine import PaperTradingState

        state = PaperTradingState()
        d = state.to_dict()
        restored = PaperTradingState.from_dict(d)

        assert restored.cash == 0.0
        assert restored.positions == {}
        assert restored.equity_curve.empty


# ============================================================================
# 2. core/reporter.py — generate_multi_strategy_report
# ============================================================================

class TestReporterMultiStrategy:
    """Cover the .ffill() changes on lines 668 and 695."""

    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d)

    def test_multi_strategy_report_without_benchmark(
        self, temp_dir, strategy_results
    ):
        """Line 668: strategy equity reindex + ffill is exercised."""
        reporter = Reporter(output_dir=temp_dir)

        results_plain = {
            name: data['result'] for name, data in strategy_results.items()
        }
        allocation = {name: 0.5 for name in results_plain}

        html = reporter.generate_multi_strategy_report(
            results=results_plain,
            capital_allocation=allocation,
            total_capital=100_000,
            save_path=None,
            auto_open=False,
        )

        assert isinstance(html, str)
        assert 'Strategy_A' in html

    def test_multi_strategy_report_with_benchmark(
        self, temp_dir, strategy_results, benchmark_equity
    ):
        """Line 695: benchmark reindex + ffill + bfill is exercised."""
        reporter = Reporter(output_dir=temp_dir)

        results_plain = {
            name: data['result'] for name, data in strategy_results.items()
        }
        allocation = {name: 0.5 for name in results_plain}

        html = reporter.generate_multi_strategy_report(
            results=results_plain,
            capital_allocation=allocation,
            total_capital=100_000,
            benchmark_equity=benchmark_equity,
            benchmark_name='TEST_BENCH',
            save_path=None,
            auto_open=False,
        )

        assert isinstance(html, str)
        assert 'TEST_BENCH' in html


# ============================================================================
# 3. core/portfolio/portfolio_manager_v2.py — benchmark alignment
# ============================================================================

class TestPortfolioManagerV2BenchmarkAlignment:
    """Cover .reindex().ffill() on line 264."""

    def test_run_backtest_with_benchmark_data(self, dates):
        """Benchmark data is aligned and scaled when provided."""
        np.random.seed(42)

        # Simple signal: long for the entire period
        signal_df = pd.DataFrame({'Signal': 1}, index=dates)
        signals = {'TEST': signal_df}

        # Simple price data
        close = 100 * (1 + np.random.normal(0.0003, 0.01, len(dates))).cumprod()
        price_df = pd.DataFrame({
            'Open': close, 'High': close * 1.01,
            'Low': close * 0.99, 'Close': close,
            'Volume': 1_000_000,
        }, index=dates)
        prices = {'TEST': price_df}

        # Benchmark — use a subset of the dates to force reindex + ffill
        bench_dates = dates[::2]  # every other business day
        bench_values = np.linspace(100, 120, len(bench_dates))
        benchmark = pd.DataFrame({'TotalValue': bench_values}, index=bench_dates)

        pm = PortfolioManagerV2(initial_capital=100_000)
        result = pm.run_backtest(
            signals, prices,
            benchmark_data=benchmark,
            benchmark_name='BENCH',
        )

        assert result.benchmark_equity is not None
        assert result.benchmark_name == 'BENCH'
        # The benchmark should be reindexed to the full date range
        assert len(result.benchmark_equity) == len(result.equity_curve)


# ============================================================================
# 4. core/risk_dashboard.py — ffill in multi-strategy dashboard
# ============================================================================

class TestRiskDashboardMultiStrategy:
    """Cover .ffill().fillna(0) on line 166."""

    def test_generate_multi_strategy_risk_dashboard(
        self, strategy_results, benchmark_equity
    ):
        """Exercise the combined equity ffill path."""
        tmp = tempfile.mkdtemp()
        try:
            dashboard = RiskDashboard(output_dir=tmp)

            results_plain = {
                name: data['result'] for name, data in strategy_results.items()
            }
            allocation = {'Strategy_A': 0.6, 'Strategy_B': 0.4}

            save_path = str(Path(tmp) / 'risk.html')

            dashboard.generate_multi_strategy_risk_dashboard(
                results=results_plain,
                capital_allocation=allocation,
                total_capital=100_000,
                save_path=save_path,
                auto_open=False,
            )

            assert Path(save_path).exists()
            content = Path(save_path).read_text()
            assert 'Risk' in content or 'risk' in content
        finally:
            shutil.rmtree(tmp)


# ============================================================================
# 5. utils/plotter.py — benchmark equity calculation
# ============================================================================

class TestPlotterBenchmarkEquity:
    """Cover .reindex().ffill() on line 496."""

    def test_calculate_benchmark_equity_with_sparse_data(
        self, strategy_results
    ):
        """Benchmark data with fewer dates forces reindex + ffill."""
        dates = list(strategy_results.values())[0]['result'].equity_curve.index

        # Sparse benchmark: only every 3rd date
        sparse_dates = dates[::3]
        np.random.seed(7)
        bench_prices = 300 * (
            1 + np.random.normal(0.0004, 0.008, len(sparse_dates))
        ).cumprod()

        bench_df = pd.DataFrame({'Close': bench_prices}, index=sparse_dates)

        plotter = PortfolioPlotter(
            strategy_results,
            benchmark_data=bench_df,
            benchmark_name='BENCH',
        )

        equity = plotter._calculate_benchmark_equity()

        assert equity is not None
        # Should match the full date range
        assert len(equity) == len(dates)
        assert not equity.isna().any()

    def test_calculate_benchmark_equity_none_when_no_benchmark(
        self, strategy_results
    ):
        """Without benchmark data the helper returns None."""
        plotter = PortfolioPlotter(strategy_results)
        assert plotter._calculate_benchmark_equity() is None
