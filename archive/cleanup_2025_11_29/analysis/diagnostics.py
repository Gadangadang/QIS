"""Diagnostic tools to understand why models underperform.

This module provides deep analysis of:
1. Signal quality: How good are the raw signals before execution?
2. Execution analysis: How much alpha is lost to execution decisions?
3. Parameter sensitivity: Which parameters matter most?
4. Regime breakdown: When does the model fail?
5. Trade anatomy: Detailed analysis of winning vs losing trades
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
from analysis.metrics import sharpe_ratio, max_drawdown


class ModelDiagnostics:
    """Comprehensive diagnostics for understanding model underperformance.
    
    Example:
        >>> results = run_walk_forward(...)
        >>> diag = ModelDiagnostics(results)
        >>> diag.signal_quality_report()  # Are signals good?
        >>> diag.execution_leakage()      # How much alpha lost to execution?
        >>> diag.regime_breakdown()       # When does it fail?
        >>> diag.save_report('logs/diagnostics.html')
    """
    
    def __init__(self, results: Dict[str, Any]):
        """Initialize diagnostics with backtest results.
        
        Args:
            results: dict from run_walk_forward() with keys:
                - stitched_equity: portfolio value over time
                - combined_returns: daily strategy returns
                - trades: all trades DataFrame
                - df: original market data
                - folds: per-fold summaries
        """
        self.results = results
        self.equity = results.get('stitched_equity', pd.Series())
        self.returns = results.get('combined_returns', pd.Series())
        self.trades = results.get('trades', pd.DataFrame())
        self.market_data = results.get('df', pd.DataFrame())
        self.folds = results.get('folds', [])
        
    def signal_quality_report(self) -> Dict[str, Any]:
        """Analyze raw signal quality before execution.
        
        Returns signal metrics like:
        - Signal Sharpe (if you had perfect execution)
        - Signal win rate (% of days signal was directionally correct)
        - Signal correlation to next-day returns
        - Signal stability across folds
        """
        if self.market_data.empty:
            return {'error': 'Market data missing'}
        
        if 'Position' not in self.market_data.columns:
            # Try to recover position from returns
            print("   ⚠ Warning: Position column not in market_data")
            return {'error': 'Position column not in market_data - signal not preserved'}
        
        df = self.market_data.copy()
        
        # Calculate next-day returns
        df['NextDayReturn'] = df['Close'].pct_change().shift(-1)
        
        # Signal directional accuracy (was signal right about next day?)
        df['SignalCorrect'] = (df['Position'] * df['NextDayReturn']) > 0
        
        # Theoretical perfect execution return (if we knew next day return)
        df['TheoreticalReturn'] = df['Position'].shift(1) * df['Close'].pct_change()
        
        metrics = {
            'signal_accuracy_pct': float(df['SignalCorrect'].mean()) if 'SignalCorrect' in df else 0,
            'signal_sharpe_theoretical': float(sharpe_ratio(df['TheoreticalReturn'].fillna(0))),
            'avg_position': float(df['Position'].abs().mean()),
            'position_changes_per_year': float((df['Position'].diff() != 0).sum() / (len(df) / 252)),
        }
        
        return metrics
    
    def execution_leakage(self) -> Dict[str, Any]:
        """Measure how much alpha is lost between signal generation and execution.
        
        Compares:
        1. Theoretical returns (perfect execution at close)
        2. Actual returns (with shift, costs, stops)
        3. Slippage and cost attribution
        """
        if self.returns.empty:
            return {'error': 'No returns data'}
        
        # Calculate theoretical vs actual
        theoretical_sharpe = self.signal_quality_report().get('signal_sharpe_theoretical', 0)
        actual_sharpe = sharpe_ratio(self.returns.fillna(0))
        
        leakage = {
            'theoretical_sharpe': theoretical_sharpe,
            'actual_sharpe': actual_sharpe,
            'sharpe_leakage': theoretical_sharpe - actual_sharpe,
            'leakage_pct': (theoretical_sharpe - actual_sharpe) / theoretical_sharpe if theoretical_sharpe != 0 else 0,
        }
        
        return leakage
    
    def regime_breakdown(self) -> pd.DataFrame:
        """Analyze performance by market regime.
        
        Returns DataFrame with performance in:
        - Bull markets (price > 200-day MA)
        - Bear markets (price < 200-day MA)
        - High volatility periods (VIX proxy > 20)
        - Low volatility periods
        - By year, by decade, by market drawdown level
        """
        if self.market_data.empty or self.returns.empty:
            return pd.DataFrame()
        
        df = self.market_data.copy()
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['BullMarket'] = df['Close'] > df['SMA200']
        
        # Volatility proxy
        df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['HighVol'] = df['Volatility'] > df['Volatility'].quantile(0.7)
        
        # Add returns
        df['StrategyReturn'] = self.returns
        
        # Group by regime
        regimes = []
        
        for name, mask in [
            ('Bull Market', df['BullMarket'] == True),
            ('Bear Market', df['BullMarket'] == False),
            ('High Vol', df['HighVol'] == True),
            ('Low Vol', df['HighVol'] == False),
        ]:
            if mask.sum() > 20:  # Need minimum data
                subset = df.loc[mask, 'StrategyReturn'].fillna(0)
                regimes.append({
                    'regime': name,
                    'n_days': len(subset),
                    'sharpe': sharpe_ratio(subset),
                    'total_return': (1 + subset).prod() - 1,
                    'max_drawdown': max_drawdown(subset),
                })
        
        return pd.DataFrame(regimes)
    
    def trade_anatomy(self) -> Dict[str, Any]:
        """Deep dive into winning vs losing trades.
        
        Returns:
        - Exit reason distribution
        - Hold period analysis (wins vs losses)
        - Entry/exit timing analysis
        - Side bias (long vs short performance)
        """
        if self.trades.empty:
            return {'error': 'No trades data'}
        
        trades = self.trades.copy()
        
        # Add hold period
        trades['hold_days'] = (pd.to_datetime(trades['exit_date']) - 
                               pd.to_datetime(trades['entry_date'])).dt.days
        
        wins = trades[trades['pnl_pct'] > 0]
        losses = trades[trades['pnl_pct'] <= 0]
        
        anatomy = {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades) if len(trades) > 0 else 0,
            
            # Exit reasons
            'exit_reasons': trades['exit_reason'].value_counts().to_dict() if 'exit_reason' in trades else {},
            
            # Hold periods
            'avg_hold_days_wins': float(wins['hold_days'].mean()) if not wins.empty else 0,
            'avg_hold_days_losses': float(losses['hold_days'].mean()) if not losses.empty else 0,
            
            # Side analysis
            'long_win_rate': float((trades[trades['side'] == 'long']['pnl_pct'] > 0).mean()) if 'side' in trades else 0,
            'short_win_rate': float((trades[trades['side'] == 'short']['pnl_pct'] > 0).mean()) if 'side' in trades else 0,
            
            # Size analysis
            'avg_win_size': float(wins['pnl_pct'].mean()) if not wins.empty else 0,
            'avg_loss_size': float(losses['pnl_pct'].mean()) if not losses.empty else 0,
            'largest_win': float(wins['pnl_pct'].max()) if not wins.empty else 0,
            'largest_loss': float(losses['pnl_pct'].min()) if not losses.empty else 0,
        }
        
        return anatomy
    
    def parameter_sensitivity(self, param_name: str, param_values: List) -> pd.DataFrame:
        """Test sensitivity to a specific parameter.
        
        Args:
            param_name: Parameter to test (e.g., 'lookback', 'threshold')
            param_values: List of values to test
            
        Returns:
            DataFrame with Sharpe, return, etc. for each param value
        """
        # This would require re-running backtests - placeholder for now
        return pd.DataFrame({
            'param_value': param_values,
            'sharpe': [np.nan] * len(param_values),
            'note': ['Requires re-run with different params'] * len(param_values)
        })
    
    def fold_stability(self) -> pd.DataFrame:
        """Analyze consistency across walk-forward folds.
        
        Returns:
            DataFrame showing per-fold metrics and standard deviation
        """
        if not self.folds:
            return pd.DataFrame()
        
        fold_df = pd.DataFrame(self.folds)
        
        # Add stability metrics
        stats = pd.DataFrame({
            'metric': ['sharpe', 'fold_return_pct', 'max_drawdown', 'n_trades'],
            'mean': [fold_df[m].mean() for m in ['sharpe', 'fold_return_pct', 'max_drawdown', 'n_trades']],
            'std': [fold_df[m].std() for m in ['sharpe', 'fold_return_pct', 'max_drawdown', 'n_trades']],
            'min': [fold_df[m].min() for m in ['sharpe', 'fold_return_pct', 'max_drawdown', 'n_trades']],
            'max': [fold_df[m].max() for m in ['sharpe', 'fold_return_pct', 'max_drawdown', 'n_trades']],
        })
        
        return stats
    
    def summary_report(self) -> str:
        """Generate comprehensive text report with all diagnostics."""
        report = []
        report.append("=" * 70)
        report.append("MODEL DIAGNOSTICS REPORT")
        report.append("=" * 70)
        
        # 1. Signal Quality
        report.append("\n1. SIGNAL QUALITY")
        report.append("-" * 70)
        signal_qual = self.signal_quality_report()
        if 'error' not in signal_qual:
            report.append(f"Signal Accuracy:           {signal_qual.get('signal_accuracy_pct', 0):.1%}")
            report.append(f"Theoretical Sharpe:        {signal_qual.get('signal_sharpe_theoretical', 0):.3f}")
            report.append(f"Position Changes/Year:     {signal_qual.get('position_changes_per_year', 0):.1f}")
        else:
            report.append(f"Error: {signal_qual['error']}")
        
        # 2. Execution Leakage
        report.append("\n2. EXECUTION LEAKAGE")
        report.append("-" * 70)
        leakage = self.execution_leakage()
        if 'error' not in leakage:
            report.append(f"Theoretical Sharpe:        {leakage.get('theoretical_sharpe', 0):+.3f}")
            report.append(f"Actual Sharpe:             {leakage.get('actual_sharpe', 0):+.3f}")
            report.append(f"Leakage:                   {leakage.get('sharpe_leakage', 0):+.3f} ({leakage.get('leakage_pct', 0):.1%})")
        else:
            report.append(f"Error: {leakage['error']}")
        
        # 3. Regime Breakdown
        report.append("\n3. REGIME BREAKDOWN")
        report.append("-" * 70)
        regimes = self.regime_breakdown()
        if not regimes.empty:
            for _, row in regimes.iterrows():
                report.append(f"{row['regime']:15s}: Sharpe {row['sharpe']:+.3f} | Return {row['total_return']:+.2%} | Days {int(row['n_days']):,}")
        else:
            report.append("No regime data available")
        
        # 4. Trade Anatomy
        report.append("\n4. TRADE ANATOMY")
        report.append("-" * 70)
        anatomy = self.trade_anatomy()
        if 'error' not in anatomy:
            report.append(f"Total Trades:              {anatomy.get('total_trades', 0)}")
            report.append(f"Win Rate:                  {anatomy.get('win_rate', 0):.1%}")
            report.append(f"Avg Hold (Wins):           {anatomy.get('avg_hold_days_wins', 0):.1f} days")
            report.append(f"Avg Hold (Losses):         {anatomy.get('avg_hold_days_losses', 0):.1f} days")
            report.append(f"Long Win Rate:             {anatomy.get('long_win_rate', 0):.1%}")
            report.append(f"Short Win Rate:            {anatomy.get('short_win_rate', 0):.1%}")
            report.append(f"\nExit Reasons:")
            for reason, count in anatomy.get('exit_reasons', {}).items():
                report.append(f"  {reason or 'signal_exit':15s}: {count:4d} trades")
        else:
            report.append(f"Error: {anatomy['error']}")
        
        # 5. Fold Stability
        report.append("\n5. FOLD STABILITY")
        report.append("-" * 70)
        stability = self.fold_stability()
        if not stability.empty:
            for _, row in stability.iterrows():
                report.append(f"{row['metric']:20s}: mean={row['mean']:7.3f} | std={row['std']:6.3f} | range=[{row['min']:7.3f}, {row['max']:7.3f}]")
        else:
            report.append("No fold data available")
        
        report.append("\n" + "=" * 70)
        return "\n".join(report)
    
    def save_report(self, filepath: str):
        """Save comprehensive diagnostics report to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(self.summary_report())
        
        print(f"Diagnostics report saved to: {filepath}")
