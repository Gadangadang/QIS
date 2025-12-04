"""
Comparison test: Orchestrator vs Manual approach.

This test proves that BacktestOrchestrator produces identical results
to the manual notebook-style approach.

Run with: pytest tests/test_orchestrator_comparison.py -v -s
"""

import pytest
import pandas as pd
import numpy as np

from core.backtest_orchestrator import BacktestOrchestrator
from core.multi_asset_loader import load_assets
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2
from signals.momentum import MomentumSignalV2


class TestOrchestratorVsManual:
    """Compare orchestrator results with manual approach."""
    
    @pytest.fixture
    def test_data(self):
        """Create consistent test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')  # 1 year
        
        return {
            'ES': pd.DataFrame({
                'Open': 4500 + np.cumsum(np.random.randn(252)) * 10,
                'High': 4520 + np.cumsum(np.random.randn(252)) * 10,
                'Low': 4480 + np.cumsum(np.random.randn(252)) * 10,
                'Close': 4500 + np.cumsum(np.random.randn(252)) * 10,
                'Volume': 1000000 + np.random.randint(-100000, 100000, 252)
            }, index=dates),
            'NQ': pd.DataFrame({
                'Open': 15000 + np.cumsum(np.random.randn(252)) * 50,
                'High': 15020 + np.cumsum(np.random.randn(252)) * 50,
                'Low': 14980 + np.cumsum(np.random.randn(252)) * 50,
                'Close': 15000 + np.cumsum(np.random.randn(252)) * 50,
                'Volume': 800000 + np.random.randint(-80000, 80000, 252)
            }, index=dates)
        }
    
    def test_single_strategy_same_results(self, test_data):
        """Test that orchestrator produces same results as manual approach."""
        
        # ============================================================
        # MANUAL APPROACH (old notebook style)
        # ============================================================
        print("\n" + "="*80)
        print("MANUAL APPROACH (Old Notebook Style)")
        print("="*80)
        
        # Generate signal manually
        signal_generator = MomentumSignalV2(lookback=20)
        manual_signal = signal_generator.generate(test_data['ES'].copy())
        
        # Create portfolio manager manually
        pm_manual = PortfolioManagerV2(
            initial_capital=100000,
            risk_per_trade=0.02,
            max_position_size=1.0,
            transaction_cost_bps=3.0
        )
        
        # Run backtest manually
        manual_result = pm_manual.run_backtest(
            signals={'ES': manual_signal[['Signal']]},
            prices={'ES': test_data['ES']}
        )
        
        print(f"Final Equity:  ${manual_result.final_equity:,.2f}")
        print(f"Total Return:  {manual_result.total_return:.4f}")
        print(f"Sharpe Ratio:  {manual_result.metrics['Sharpe Ratio']:.4f}")
        print(f"Max Drawdown:  {manual_result.metrics['Max Drawdown']:.4f}")
        print(f"Total Trades:  {manual_result.metrics['Total Trades']}")
        
        # ============================================================
        # ORCHESTRATOR APPROACH (new clean style)
        # ============================================================
        print("\n" + "="*80)
        print("ORCHESTRATOR APPROACH (New Clean Style)")
        print("="*80)
        
        # Create orchestrator
        orchestrator = BacktestOrchestrator()
        
        # Set data (simulating load_data)
        orchestrator.prices = test_data
        orchestrator._data_loaded = True
        
        # Add strategy, generate signals, run backtest (all in one flow)
        orchestrator.add_strategy(
            name='Momentum',
            signal_generator=MomentumSignalV2(lookback=20),
            assets=['ES'],
            capital=100000,
            risk_per_trade=0.02,
            transaction_cost_bps=3.0
        )
        
        orchestrator.generate_signals(verbose=False)
        results = orchestrator.run_backtests(verbose=False)
        
        orch_result = results['Momentum']['result']
        
        print(f"Final Equity:  ${orch_result.final_equity:,.2f}")
        print(f"Total Return:  {orch_result.total_return:.4f}")
        print(f"Sharpe Ratio:  {orch_result.metrics['Sharpe Ratio']:.4f}")
        print(f"Max Drawdown:  {orch_result.metrics['Max Drawdown']:.4f}")
        print(f"Total Trades:  {orch_result.metrics['Total Trades']}")
        
        # ============================================================
        # COMPARISON - RESULTS MUST BE IDENTICAL
        # ============================================================
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        
        # Compare final equity (should be exactly the same)
        equity_diff = abs(manual_result.final_equity - orch_result.final_equity)
        print(f"Final Equity Difference: ${equity_diff:.2f}")
        assert equity_diff < 0.01, "Final equity should be identical"
        
        # Compare total return
        return_diff = abs(manual_result.total_return - orch_result.total_return)
        print(f"Total Return Difference: {return_diff:.6f}")
        assert return_diff < 1e-10, "Total return should be identical"
        
        # Compare Sharpe ratio
        sharpe_diff = abs(manual_result.metrics['Sharpe Ratio'] - orch_result.metrics['Sharpe Ratio'])
        print(f"Sharpe Ratio Difference: {sharpe_diff:.6f}")
        assert sharpe_diff < 1e-10, "Sharpe ratio should be identical"
        
        # Compare drawdown
        dd_diff = abs(manual_result.metrics['Max Drawdown'] - orch_result.metrics['Max Drawdown'])
        print(f"Max Drawdown Difference: {dd_diff:.6f}")
        assert dd_diff < 1e-10, "Max drawdown should be identical"
        
        # Compare trade count
        trade_diff = abs(manual_result.metrics['Total Trades'] - orch_result.metrics['Total Trades'])
        print(f"Trade Count Difference:  {trade_diff}")
        assert trade_diff == 0, "Trade count should be identical"
        
        print("\n✅ ALL METRICS MATCH PERFECTLY!")
        print("="*80)
    
    def test_multi_strategy_same_results(self, test_data):
        """Test multi-strategy backtests produce same results."""
        
        print("\n" + "="*80)
        print("MULTI-STRATEGY COMPARISON")
        print("="*80)
        
        # ============================================================
        # MANUAL APPROACH
        # ============================================================
        print("\nManual Approach:")
        
        # Strategy 1: Momentum on ES
        sig1 = MomentumSignalV2(lookback=20).generate(test_data['ES'].copy())
        pm1 = PortfolioManagerV2(
            initial_capital=50000, 
            risk_per_trade=0.02,
            max_position_size=1.0,
            transaction_cost_bps=3.0
        )
        result1_manual = pm1.run_backtest(
            signals={'ES': sig1[['Signal']]},
            prices={'ES': test_data['ES']}
        )
        
        # Strategy 2: Momentum on NQ
        sig2 = MomentumSignalV2(lookback=30).generate(test_data['NQ'].copy())
        pm2 = PortfolioManagerV2(
            initial_capital=50000, 
            risk_per_trade=0.02,
            max_position_size=1.0,
            transaction_cost_bps=3.0
        )
        result2_manual = pm2.run_backtest(
            signals={'NQ': sig2[['Signal']]},
            prices={'NQ': test_data['NQ']}
        )
        
        manual_total = result1_manual.final_equity + result2_manual.final_equity
        print(f"  Strategy 1 Final: ${result1_manual.final_equity:,.2f}")
        print(f"  Strategy 2 Final: ${result2_manual.final_equity:,.2f}")
        print(f"  Portfolio Total:  ${manual_total:,.2f}")
        
        # ============================================================
        # ORCHESTRATOR APPROACH
        # ============================================================
        print("\nOrchestrator Approach:")
        
        orchestrator = BacktestOrchestrator()
        orchestrator.prices = test_data
        orchestrator._data_loaded = True
        
        orchestrator.add_strategy('Strategy1', MomentumSignalV2(lookback=20), ['ES'], 50000)
        orchestrator.add_strategy('Strategy2', MomentumSignalV2(lookback=30), ['NQ'], 50000)
        
        orchestrator.generate_signals(verbose=False)
        results = orchestrator.run_backtests(verbose=False)
        
        orch_total = (results['Strategy1']['result'].final_equity + 
                      results['Strategy2']['result'].final_equity)
        
        print(f"  Strategy 1 Final: ${results['Strategy1']['result'].final_equity:,.2f}")
        print(f"  Strategy 2 Final: ${results['Strategy2']['result'].final_equity:,.2f}")
        print(f"  Portfolio Total:  ${orch_total:,.2f}")
        
        # Compare
        diff = abs(manual_total - orch_total)
        print(f"\nDifference: ${diff:.2f}")
        assert diff < 0.01
        
        print("✅ Multi-strategy results match!")
    
    def test_orchestrator_is_cleaner(self):
        """Demonstrate that orchestrator code is much cleaner."""
        
        print("\n" + "="*80)
        print("CODE COMPARISON")
        print("="*80)
        
        manual_lines = """
# MANUAL APPROACH (Typical notebook code):

# 1. Load data
prices = load_assets(['ES', 'NQ'], start_date='2020-01-01')

# 2. Generate signals
signal_gen = MomentumSignalV2(lookback=20)
signals_es = signal_gen.generate(prices['ES'].copy())
signals_nq = signal_gen.generate(prices['NQ'].copy())

# 3. Run backtests
pm_es = PortfolioManagerV2(initial_capital=50000, risk_per_trade=0.02, max_position_size=1.0)
result_es = pm_es.run_backtest(
    signals={'ES': signals_es[['Signal']]},
    prices={'ES': prices['ES']}
)

pm_nq = PortfolioManagerV2(initial_capital=50000, risk_per_trade=0.02, max_position_size=1.0)
result_nq = pm_nq.run_backtest(
    signals={'NQ': signals_nq[['Signal']]},
    prices={'NQ': prices['NQ']}
)

# 4. Format results
print(f"ES Return: {result_es.total_return:.2%}")
print(f"NQ Return: {result_nq.total_return:.2%}")

Total lines: ~20+
"""
        
        orchestrator_lines = """
# ORCHESTRATOR APPROACH (Clean and intuitive):

orchestrator = BacktestOrchestrator()

(orchestrator
    .load_data(['ES', 'NQ'], start_date='2020-01-01')
    .add_strategy('ES_Momentum', MomentumSignalV2(lookback=20), ['ES'], 50000)
    .add_strategy('NQ_Momentum', MomentumSignalV2(lookback=20), ['NQ'], 50000)
    .generate_signals()
    .run_backtests())

orchestrator.print_summary()

Total lines: ~8
"""
        
        print("\nMANUAL APPROACH:")
        print(manual_lines)
        
        print("\nORCHESTRATOR APPROACH:")
        print(orchestrator_lines)
        
        print("\n📊 Orchestrator reduces code by ~60%!")
        print("✅ More readable, less error-prone, easier to maintain")
        print("="*80)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
