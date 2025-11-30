"""
Paper Trading Engine

⚠️ DEPRECATED: This module uses the old portfolio_manager architecture.
    Needs refactoring to work with core.portfolio.portfolio_manager_v2.
    Currently maintained for reference but not used in active workflows.

Orchestrates paper trading workflows including:
- State persistence across daily runs
- Incremental updates with new data
- Performance comparison vs backtest
- Portfolio status tracking
- Daily reporting

TODO: Refactor to use:
  - core.portfolio.portfolio_manager_v2.PortfolioManagerV2
  - core.portfolio.risk_manager.RiskManager
  - core.portfolio.backtest_result.BacktestResult
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pickle
import warnings

# Using old architecture - see TODO above for migration path
from archive.cleanup_2025_11_30.old_core_duplicates.portfolio_manager import PortfolioManager, PortfolioConfig, BacktestResult
from archive.cleanup_2025_11_30.old_core_duplicates.risk_manager import RiskManager


class PaperTradingState:
    """Container for paper trading state that persists between runs"""
    
    def __init__(self):
        self.positions: Dict = {}  # Current positions {ticker: {shares: N, ...}}
        self.cash: float = 0.0
        self.equity_curve: pd.DataFrame = pd.DataFrame()  # Historical equity values
        self.trades: pd.DataFrame = pd.DataFrame()  # All executed trades
        self.last_update: Optional[datetime] = None
        self.initial_capital: float = 0.0
        self.backtest_result: Optional[BacktestResult] = None
        self.backtest_equity: Optional[pd.DataFrame] = None
        self.backtest_trades: Optional[pd.DataFrame] = None
        
    def to_dict(self) -> Dict:
        """Serialize state to dictionary"""
        return {
            'positions': self.positions,
            'cash': self.cash,
            'equity_curve': self.equity_curve.to_dict() if not self.equity_curve.empty else {},
            'trades': self.trades.to_dict() if not self.trades.empty else {},
            'last_update': self.last_update,
            'initial_capital': self.initial_capital,
            'backtest_result': self.backtest_result,
            'backtest_equity': self.backtest_equity.to_dict() if self.backtest_equity is not None else None,
            'backtest_trades': self.backtest_trades.to_dict() if self.backtest_trades is not None else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PaperTradingState':
        """Deserialize state from dictionary"""
        state = cls()
        state.positions = data.get('positions', {})
        state.cash = data.get('cash', 0.0)
        
        equity_dict = data.get('equity_curve', {})
        state.equity_curve = pd.DataFrame.from_dict(equity_dict) if equity_dict else pd.DataFrame()
        
        trades_dict = data.get('trades', {})
        state.trades = pd.DataFrame.from_dict(trades_dict) if trades_dict else pd.DataFrame()
        
        state.last_update = data.get('last_update')
        state.initial_capital = data.get('initial_capital', 0.0)
        state.backtest_result = data.get('backtest_result')
        
        backtest_equity_dict = data.get('backtest_equity')
        state.backtest_equity = pd.DataFrame.from_dict(backtest_equity_dict) if backtest_equity_dict else None
        
        backtest_trades_dict = data.get('backtest_trades')
        state.backtest_trades = pd.DataFrame.from_dict(backtest_trades_dict) if backtest_trades_dict else None
        
        return state


class PaperTradingEngine:
    """
    Manages paper trading workflow:
    - Maintains state across runs
    - Processes incremental updates
    - Compares live vs backtest performance
    - Generates reports
    """
    
    def __init__(
        self,
        config: PortfolioConfig,
        backtest_result: Optional[BacktestResult] = None,
        backtest_equity: Optional[pd.DataFrame] = None,
        backtest_trades: Optional[pd.DataFrame] = None,
        state: Optional[PaperTradingState] = None
    ):
        """
        Initialize paper trading engine
        
        Args:
            config: Portfolio configuration
            backtest_result: Reference backtest result for comparison
            backtest_equity: Backtest equity curve
            backtest_trades: Backtest trades
            state: Existing state to resume from (optional)
        """
        self.config = config
        self.state = state if state is not None else PaperTradingState()
        
        # Store backtest reference
        if backtest_result is not None:
            self.state.backtest_result = backtest_result
            self.state.backtest_equity = backtest_equity
            self.state.backtest_trades = backtest_trades
            self.state.initial_capital = config.initial_capital
    
    def initialize(
        self,
        prices_dict: Dict[str, pd.DataFrame],
        signals_dict: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None
    ) -> None:
        """
        Initialize paper trading from scratch
        
        Args:
            prices_dict: Price data for all assets
            signals_dict: Signal data for all assets
            start_date: Date to start paper trading (default: first date in data)
        """
        from core.portfolio_manager import run_multi_asset_backtest
        
        # Filter data to start_date if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            prices_dict = {
                ticker: df[df.index >= start_dt].copy()
                for ticker, df in prices_dict.items()
            }
            signals_dict = {
                ticker: df[df.index >= start_dt].copy()
                for ticker, df in signals_dict.items()
            }
        
        # Run initial backtest to get starting state
        # IMPORTANT: Use a fresh risk manager for live trading
        # We don't want to inherit drawdown stops from the reference backtest
        from core.portfolio.risk_manager import RiskManager
        live_risk_manager = RiskManager(self.config.risk_manager.config)
        
        # Create a fresh config with the new risk manager
        live_config = PortfolioConfig(
            initial_capital=self.config.initial_capital,
            rebalance_threshold=self.config.rebalance_threshold,
            transaction_cost_bps=self.config.transaction_cost_bps,
            risk_manager=live_risk_manager,
            rejection_policy=self.config.rejection_policy
        )
        
        result, equity, trades = run_multi_asset_backtest(
            signals_dict=signals_dict,
            prices_dict=prices_dict,
            config=live_config,
            return_pm=False
        )
        
        # Initialize state from first run
        self.state.equity_curve = equity
        self.state.trades = trades
        
        if len(equity) > 0:
            last_row = equity.iloc[-1]
            self.state.cash = last_row['Cash']
            self.state.positions = last_row['Positions']
            self.state.last_update = pd.to_datetime(last_row['Date'])
        
        self.state.initial_capital = self.config.initial_capital
    
    def update(
        self,
        prices_dict: Dict[str, pd.DataFrame],
        signals_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Update paper trading with new data
        
        Args:
            prices_dict: Updated price data (should include history for indicators)
            signals_dict: Updated signal data
            
        Returns:
            Tuple of (updated_equity, new_trades)
        """
        from core.portfolio.portfolio_manager import run_multi_asset_backtest
        from core.portfolio.risk_manager import RiskManager
        
        # Use a fresh risk manager for each update to avoid carrying over stale violations
        live_risk_manager = RiskManager(self.config.risk_manager.config)
        
        # Create a fresh config with the new risk manager
        live_config = PortfolioConfig(
            initial_capital=self.config.initial_capital,
            rebalance_threshold=self.config.rebalance_threshold,
            transaction_cost_bps=self.config.transaction_cost_bps,
            risk_manager=live_risk_manager,
            rejection_policy=self.config.rejection_policy
        )
        
        # Run backtest on full dataset
        result, equity_full, trades_full = run_multi_asset_backtest(
            signals_dict=signals_dict,
            prices_dict=prices_dict,
            config=live_config,
            return_pm=False
        )
        
        # Update state with latest results
        self.state.equity_curve = equity_full
        self.state.trades = trades_full
        
        if len(equity_full) > 0:
            last_row = equity_full.iloc[-1]
            self.state.cash = last_row['Cash']
            self.state.positions = last_row['Positions']
            self.state.last_update = pd.to_datetime(last_row['Date'])
        
        return equity_full, trades_full
    
    def get_portfolio_status(self, prices_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Get current portfolio status
        
        Args:
            prices_dict: Current price data for valuation
            
        Returns:
            Dictionary with portfolio details
        """
        if len(self.state.equity_curve) == 0:
            return {
                'status': 'not_initialized',
                'message': 'Paper trading not initialized'
            }
        
        last_state = self.state.equity_curve.iloc[-1]
        last_date = pd.to_datetime(last_state['Date'])
        total_value = last_state['TotalValue']
        cash = last_state['Cash']
        positions_dict = last_state['Positions']
        
        # Build position details
        position_details = []
        total_invested = 0.0
        
        if positions_dict and len(positions_dict) > 0:
            for ticker, position_info in positions_dict.items():
                # Handle both dict and simple number formats
                if isinstance(position_info, dict):
                    # Check both lowercase and uppercase keys for compatibility
                    shares = position_info.get('Shares', position_info.get('shares', 0))
                else:
                    shares = position_info
                
                if shares > 0:
                    # Get current price
                    if ticker in prices_dict and len(prices_dict[ticker]) > 0:
                        current_price = prices_dict[ticker]['Close'].iloc[-1]
                        position_value = shares * current_price
                        total_invested += position_value
                        
                        # Find entry from trades
                        entry_price = None
                        entry_date = None
                        if not self.state.trades.empty:
                            ticker_trades = self.state.trades[self.state.trades['Ticker'] == ticker]
                            if len(ticker_trades) > 0:
                                last_trade = ticker_trades.iloc[-1]
                                entry_price = last_trade['Price']
                                entry_date = pd.to_datetime(last_trade['Date'])
                        
                        position_details.append({
                            'ticker': ticker,
                            'shares': shares,
                            'current_price': current_price,
                            'position_value': position_value,
                            'entry_price': entry_price,
                            'entry_date': entry_date,
                            'unrealized_pnl': (current_price - entry_price) * shares if entry_price else None,
                            'unrealized_pct': (current_price / entry_price - 1) * 100 if entry_price else None
                        })
        
        return {
            'status': 'active',
            'as_of_date': last_date,
            'total_value': total_value,
            'cash': cash,
            'invested': total_invested,
            'num_positions': len(position_details),
            'positions': position_details,
            'initial_capital': self.state.initial_capital,
            'total_return': (total_value / self.state.initial_capital - 1) if self.state.initial_capital > 0 else 0,
        }
    
    def get_performance_comparison(self, live_start_date: str) -> Dict:
        """
        Compare live vs backtest performance
        
        Args:
            live_start_date: Date when live trading started
            
        Returns:
            Dictionary with comparison metrics
        """
        live_start_dt = pd.to_datetime(live_start_date)
        
        # Filter to live period
        equity_live = self.state.equity_curve
        if not equity_live.empty:
            equity_live = equity_live[pd.to_datetime(equity_live['Date']) >= live_start_dt].copy()
        
        trades_live = self.state.trades
        if not trades_live.empty:
            trades_live = trades_live[pd.to_datetime(trades_live['Date']) >= live_start_dt].copy()
        
        # Calculate live metrics
        live_metrics = {}
        if len(equity_live) > 1:
            start_value = equity_live['TotalValue'].iloc[0]
            end_value = equity_live['TotalValue'].iloc[-1]
            live_return = (end_value - start_value) / start_value
            live_pnl = end_value - start_value
            
            # Calculate max drawdown for live period
            equity_values = equity_live['TotalValue'].values
            cummax = pd.Series(equity_values).expanding().max()
            drawdowns = (equity_values - cummax) / cummax
            max_drawdown = drawdowns.min()
            
            # Calculate Sharpe ratio for live period (if enough data)
            sharpe = None
            if len(equity_live) >= 30:  # Need at least 30 days
                returns = pd.Series(equity_values).pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
            
            live_metrics = {
                'start_value': start_value,
                'end_value': end_value,
                'total_return': live_return,
                'pnl': live_pnl,
                'max_drawdown': max_drawdown,
                'sharpe': sharpe,
                'num_trades': len(trades_live),
                'num_days': len(equity_live),
            }
        
        # Get backtest metrics
        backtest_metrics = {}
        if self.state.backtest_result is not None:
            bt_metrics = self.state.backtest_result.calculate_metrics()
            backtest_metrics = {
                'total_return': bt_metrics.get('Total Return', 0),
                'cagr': bt_metrics.get('CAGR', 0),
                'sharpe': bt_metrics.get('Sharpe Ratio', 0),
                'max_drawdown': bt_metrics.get('Max Drawdown', 0),
                'num_trades': len(self.state.backtest_trades) if self.state.backtest_trades is not None else 0,
                'num_days': len(self.state.backtest_equity) if self.state.backtest_equity is not None else 0,
            }
        
        return {
            'live': live_metrics,
            'backtest': backtest_metrics,
            'live_start_date': live_start_date
        }
    
    def generate_daily_report(
        self,
        prices_dict: Dict[str, pd.DataFrame],
        signals_dict: Dict[str, pd.DataFrame],
        live_start_date: str
    ) -> str:
        """
        Generate formatted daily report
        
        Args:
            prices_dict: Current price data
            signals_dict: Current signals
            live_start_date: Date when live trading started
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("PAPER TRADING DAILY REPORT")
        report_lines.append("="*60)
        
        # Portfolio status
        status = self.get_portfolio_status(prices_dict)
        
        if status['status'] == 'not_initialized':
            report_lines.append("\n⚠️  Paper trading not initialized")
            return "\n".join(report_lines)
        
        report_lines.append(f"\nAs of: {status['as_of_date'].date()}")
        report_lines.append(f"\nPortfolio Value: ${status['total_value']:,.2f}")
        report_lines.append(f"  Cash: ${status['cash']:,.2f}")
        report_lines.append(f"  Invested: ${status['invested']:,.2f}")
        report_lines.append(f"  Total Return: {status['total_return']:.2%}")
        
        # Positions
        if status['num_positions'] > 0:
            report_lines.append(f"\n📍 Open Positions: {status['num_positions']}")
            for pos in status['positions']:
                report_lines.append(f"\n  {pos['ticker']}:")
                report_lines.append(f"    Shares: {pos['shares']:.0f}")
                report_lines.append(f"    Current: ${pos['current_price']:.2f}")
                report_lines.append(f"    Value: ${pos['position_value']:,.2f}")
                if pos['entry_price']:
                    report_lines.append(f"    Entry: ${pos['entry_price']:.2f} on {pos['entry_date'].date()}")
                    report_lines.append(f"    Unrealized P&L: ${pos['unrealized_pnl']:,.2f} ({pos['unrealized_pct']:+.2f}%)")
        else:
            report_lines.append(f"\n💰 No open positions - 100% cash")
        
        # Current signals
        report_lines.append(f"\n📊 Current Signals:")
        for ticker in signals_dict.keys():
            if len(signals_dict[ticker]) > 0:
                latest_signal = signals_dict[ticker]['Signal'].iloc[-1]
                signal_text = "LONG" if latest_signal == 1 else "FLAT"
                report_lines.append(f"  {ticker}: {signal_text}")
        
        # Performance comparison
        comparison = self.get_performance_comparison(live_start_date)
        if comparison['live']:
            report_lines.append(f"\n📈 Live Performance (from {live_start_date}):")
            report_lines.append(f"  Return: {comparison['live']['total_return']:.2%}")
            report_lines.append(f"  P&L: ${comparison['live']['pnl']:,.2f}")
            report_lines.append(f"  Trades: {comparison['live']['num_trades']}")
            report_lines.append(f"  Days: {comparison['live']['num_days']}")
        
        if comparison['backtest']:
            report_lines.append(f"\n📊 Backtest Reference:")
            report_lines.append(f"  Total Return: {comparison['backtest']['total_return']:.2%}")
            report_lines.append(f"  CAGR: {comparison['backtest']['cagr']:.2%}")
            report_lines.append(f"  Sharpe: {comparison['backtest']['sharpe']:.3f}")
        
        report_lines.append("\n" + "="*60)
        
        return "\n".join(report_lines)
    
    def save_state(self, filepath: str) -> None:
        """Save state to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.state, f)
    
    @classmethod
    def load_state(cls, filepath: str, config: PortfolioConfig) -> 'PaperTradingEngine':
        """Load state from disk"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        return cls(config=config, state=state)
    
    def export_state_summary(self) -> Dict:
        """Export state summary (for debugging/monitoring)"""
        return {
            'last_update': self.state.last_update,
            'num_positions': len(self.state.positions) if self.state.positions else 0,
            'cash': self.state.cash,
            'equity_curve_length': len(self.state.equity_curve),
            'total_trades': len(self.state.trades),
            'initial_capital': self.state.initial_capital,
            'has_backtest_reference': self.state.backtest_result is not None,
        }
