"""
PortfolioManagerV2 - Refactored portfolio manager with clean separation of concerns.

Responsibilities:
- Orchestrate backtesting workflow
- Coordinate between Portfolio, RiskManager, ExecutionEngine
- Generate BacktestResult
"""

from typing import Dict, Optional
import pandas as pd
from pathlib import Path
from datetime import datetime

from .portfolio import Portfolio
from .risk_manager import RiskManager, RiskConfig
from .execution_engine import ExecutionEngine, ExecutionConfig
from .backtest_result import BacktestResult
from .position_sizers import PositionSizer, FixedFractionalSizer


class PortfolioManagerV2:
    """
    Orchestrates portfolio backtesting with risk management.
    
    Clean API for running backtests with proper risk controls and execution modeling.
    
    Example:
        pm = PortfolioManagerV2(
            initial_capital=100000,
            risk_per_trade=0.02,
            max_position_size=0.20,
            transaction_cost_bps=3.0,
            stop_loss_pct=0.10
        )
        
        result = pm.run_backtest(signals, prices)
        result.print_summary()
        result.plot_equity_curve()
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.20,
        transaction_cost_bps: float = 3.0,
        slippage_bps: float = 0.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        rebalance_threshold: Optional[float] = None,
        rebalance_frequency: str = 'never',
        position_sizer: Optional[PositionSizer] = None,
        risk_log_path: Optional[str] = None
    ):
        """
        Initialize portfolio manager with configuration.
        
        Args:
            initial_capital: Starting capital
            risk_per_trade: Risk per trade as decimal (e.g., 0.02 = 2%)
            max_position_size: Max position size as decimal (e.g., 0.20 = 20%)
            transaction_cost_bps: Transaction costs in basis points
            slippage_bps: Slippage in basis points
            stop_loss_pct: Stop loss percentage (e.g., 0.10 = 10%)
            take_profit_pct: Take profit percentage (e.g., 0.25 = 25%)
            rebalance_threshold: Drift threshold for rebalancing (e.g., 0.10 = 10%)
                               None = no drift-based rebalancing
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'never'
                               'never' = only drift-based rebalancing
            position_sizer: PositionSizer instance for calculating position sizes
                          If None, uses FixedFractionalSizer with max_position_size and risk_per_trade
        """
        self.initial_capital = initial_capital
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_frequency = rebalance_frequency
        self.last_rebalance_date = None
        
        # Create sub-components
        self.risk_config = RiskConfig(
            risk_per_trade=risk_per_trade,
            max_position_size=max_position_size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
        
        self.execution_config = ExecutionConfig(
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps
        )
        
        # Create position sizer if not provided
        if position_sizer is None:
            position_sizer = FixedFractionalSizer(
                max_position_pct=max_position_size,
                risk_per_trade=risk_per_trade
            )
        
        self.risk_manager = RiskManager(self.risk_config, position_sizer=position_sizer)
        self.execution_engine = ExecutionEngine(self.execution_config)
        
        # Risk rejection logging
        self.risk_log_path = risk_log_path
        self.risk_rejections = []  # Store rejections in memory
    
    def run_backtest(
        self, 
        signals: Dict[str, pd.DataFrame],
        prices: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame = None,
        benchmark_name: str = None
    ) -> BacktestResult:
        """
        Run backtest with risk management and execution simulation.
        
        Args:
            signals: Dict of ticker -> DataFrame with 'Signal' column (0 or 1)
                    Index must be DatetimeIndex
            prices: Dict of ticker -> DataFrame with OHLC data
                   Must have 'Close' column, DatetimeIndex
            benchmark_data: Optional benchmark data from BenchmarkLoader
                          DataFrame with DatetimeIndex and 'TotalValue' column (base 100)
            benchmark_name: Name of benchmark (e.g., 'SPY', 'VT')
            
        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        # Initialize portfolio
        portfolio = Portfolio(self.initial_capital)
        
        # Get all dates (union of all price data dates)
        all_dates = sorted(set().union(*[df.index for df in prices.values()]))
        
        # Track equity over time
        equity_records = []
        
        # Simulate day by day
        for date in all_dates:
            # Get current prices for this date
            current_prices = {}
            for ticker, price_df in prices.items():
                if date in price_df.index:
                    current_prices[ticker] = price_df.loc[date, 'Close']
            
            # Update portfolio with current prices
            portfolio.update_prices(current_prices)
            
            # Check risk controls on existing positions
            for ticker in list(portfolio.positions.keys()):
                position = portfolio.get_position(ticker)
                
                if position is None:
                    continue
                
                # Check stop loss
                if self.risk_manager.check_stop_loss(position):
                    if ticker in current_prices:
                        # Exit due to stop loss
                        fill_price, cost = self.execution_engine.execute_sell(
                            ticker, position.shares, current_prices[ticker]
                        )
                        portfolio.close_position(ticker, fill_price, date)
                        portfolio.cash -= cost
                    continue
                
                # Check take profit
                if self.risk_manager.check_take_profit(position):
                    if ticker in current_prices:
                        # Exit due to take profit
                        fill_price, cost = self.execution_engine.execute_sell(
                            ticker, position.shares, current_prices[ticker]
                        )
                        portfolio.close_position(ticker, fill_price, date)
                        portfolio.cash -= cost
                    continue
            
            # Process signals for each asset
            for ticker in signals.keys():
                # Skip if no data for this date
                if date not in signals[ticker].index or ticker not in current_prices:
                    continue
                
                signal = signals[ticker].loc[date, 'Signal']
                price = current_prices[ticker]
                
                has_position = portfolio.has_position(ticker)
                
                # Entry logic: signal = 1 and no position
                if signal == 1 and not has_position:
                    # Calculate position size
                    shares = self.risk_manager.calculate_position_size(
                        ticker=ticker,
                        signal=signal,
                        current_price=price,
                        portfolio_value=portfolio.total_value
                    )
                    
                    # Log if position sizing failed
                    if shares <= 0:
                        self._log_risk_rejection(
                            date=date,
                            ticker=ticker,
                            signal=signal,
                            price=price,
                            reason='Position sizer returned 0 shares',
                            portfolio_value=portfolio.total_value,
                            cash=portfolio.cash
                        )
                    
                    # Check if trade is worth executing
                    if shares > 0 and self.execution_engine.should_execute(shares, price):
                        # Execute buy order
                        fill_price, cost = self.execution_engine.execute_buy(
                            ticker, shares, price
                        )
                        
                        # Try to open position
                        try:
                            portfolio.open_position(ticker, shares, fill_price, date)
                            portfolio.cash -= cost
                        except ValueError as e:
                            # Insufficient cash or other issue, skip
                            self._log_risk_rejection(
                                date=date,
                                ticker=ticker,
                                signal=signal,
                                price=price,
                                reason=f'Portfolio rejected: {str(e)}',
                                portfolio_value=portfolio.total_value,
                                cash=portfolio.cash
                            )
                            pass
                
                # Exit logic: signal = 0 and we have position
                elif signal == 0 and has_position:
                    position = portfolio.get_position(ticker)
                    if position:
                        # Execute sell order
                        fill_price, cost = self.execution_engine.execute_sell(
                            ticker, position.shares, price
                        )
                        portfolio.close_position(ticker, fill_price, date)
                        portfolio.cash -= cost
            
            # Check for rebalancing (after signal processing)
            if self.rebalance_threshold is not None and len(portfolio.positions) > 1:
                if self._should_rebalance(date, portfolio, current_prices):
                    self._rebalance_portfolio(portfolio, current_prices, date)
            
            # Record daily equity
            equity_records.append(portfolio.get_equity_curve_point(date))
        
        # Create result DataFrames
        equity_curve = pd.DataFrame(equity_records)
        if len(equity_curve) > 0:
            equity_curve.set_index('Date', inplace=True)
        
        trades = pd.DataFrame(portfolio.closed_positions)
        
        # Prepare benchmark data if provided
        benchmark_equity = None
        if benchmark_data is not None and len(equity_curve) > 0:
            # Align benchmark to portfolio dates
            benchmark_aligned = benchmark_data.reindex(equity_curve.index, method='ffill')
            if 'TotalValue' in benchmark_aligned.columns:
                # Scale benchmark to match initial capital (maintaining DataFrame structure)
                benchmark_equity = pd.DataFrame(index=benchmark_aligned.index)
                benchmark_equity['TotalValue'] = benchmark_aligned['TotalValue'] * (self.initial_capital / 100.0)
        
        # Save risk rejection log if configured
        if self.risk_log_path and self.risk_rejections:
            self._save_risk_log()
            
        # Create and return BacktestResult
        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=self.initial_capital,
            benchmark_equity=benchmark_equity,
            benchmark_name=benchmark_name
        )
    
    def _should_rebalance(self, date: pd.Timestamp, portfolio, prices: Dict[str, float]) -> bool:
        """Check if rebalancing is needed based on drift threshold and frequency."""
        # Check frequency-based rebalancing
        if self.rebalance_frequency != 'never':
            if self.last_rebalance_date is None:
                return True
            
            days_since_rebalance = (date - self.last_rebalance_date).days
            
            if self.rebalance_frequency == 'daily' and days_since_rebalance >= 1:
                return True
            elif self.rebalance_frequency == 'weekly' and days_since_rebalance >= 7:
                return True
            elif self.rebalance_frequency == 'monthly' and days_since_rebalance >= 30:
                return True
        
        # Check drift-based rebalancing
        if self.rebalance_threshold is not None:
            allocation = portfolio.get_allocation()
            # Exclude Cash from active positions
            active_positions = [t for t, pct in allocation.items() if pct > 0 and t != 'Cash']
            
            if len(active_positions) == 0:
                return False
            
            # Target weight is equal among active positions
            target_weight = 1.0 / len(active_positions)
            
            # Check if any position drifted beyond threshold
            for ticker in active_positions:
                drift = abs(allocation[ticker] - target_weight)
                if drift > self.rebalance_threshold:
                    return True
        
        return False
    
    def _rebalance_portfolio(self, portfolio, prices: Dict[str, float], date: pd.Timestamp):
        """Rebalance portfolio to equal weights among active positions."""
        allocation = portfolio.get_allocation()
        # Exclude Cash from active positions
        active_positions = [t for t, pct in allocation.items() if pct > 0 and t != 'Cash']
        
        if len(active_positions) <= 1:
            return  # Nothing to rebalance
        
        # Target equal weight
        target_weight = 1.0 / len(active_positions)
        total_value = portfolio.total_value
        
        # Calculate target values and execute trades
        for ticker in active_positions:
            if ticker not in prices:
                continue
            
            position = portfolio.get_position(ticker)
            current_value = position.market_value
            target_value = total_value * target_weight
            
            # Skip small adjustments (< $100)
            if abs(target_value - current_value) < 100:
                continue
            
            price = prices[ticker]
            current_shares = position.shares
            target_shares = target_value / price
            shares_to_trade = target_shares - current_shares
            
            if shares_to_trade > 0:
                # Buy more shares
                fill_price, cost = self.execution_engine.execute_buy(
                    ticker, shares_to_trade, price
                )
                # Update position
                portfolio.positions[ticker].shares = target_shares
                portfolio.positions[ticker].current_price = fill_price
                portfolio.cash -= (shares_to_trade * fill_price + cost)
            else:
                # Sell shares
                fill_price, cost = self.execution_engine.execute_sell(
                    ticker, abs(shares_to_trade), price
                )
                # Update position
                portfolio.positions[ticker].shares = target_shares
                portfolio.positions[ticker].current_price = fill_price
                portfolio.cash += (abs(shares_to_trade) * fill_price - cost)
        
        self.last_rebalance_date = date
    
    def get_config_summary(self) -> str:
        """Get human-readable configuration summary."""
        lines = [
            "\n" + "="*60,
            "PORTFOLIO MANAGER CONFIGURATION",
            "="*60,
            f"Initial Capital:       ${self.initial_capital:>15,.2f}",
            f"Risk per Trade:        {self.risk_config.risk_per_trade:>15.1%}",
            f"Max Position Size:     {self.risk_config.max_position_size:>15.1%}",
            f"Transaction Cost:      {self.execution_config.transaction_cost_bps:>15.1f} bps",
            f"Slippage:              {self.execution_config.slippage_bps:>15.1f} bps",
            f"Stop Loss:             {self.risk_config.stop_loss_pct or 'None':>15}",
            f"Take Profit:           {self.risk_config.take_profit_pct or 'None':>15}",
            f"Rebalance Threshold:   {self.rebalance_threshold or 'None':>15}",
            f"Rebalance Frequency:   {self.rebalance_frequency:>15}",
            "="*60 + "\n"
        ]
        return '\n'.join(lines)
    
    def _log_risk_rejection(
        self, 
        date: pd.Timestamp, 
        ticker: str, 
        signal: float,
        price: float,
        reason: str,
        portfolio_value: float,
        cash: float
    ):
        """Log a rejected trade for later analysis."""
        self.risk_rejections.append({
            'Date': date,
            'Ticker': ticker,
            'Signal': signal,
            'Price': price,
            'Reason': reason,
            'PortfolioValue': portfolio_value,
            'Cash': cash
        })
    
    def _save_risk_log(self):
        """Save risk rejections to CSV file."""
        if not self.risk_rejections:
            return
        
        df = pd.DataFrame(self.risk_rejections)
        
        # Create directory if needed
        log_path = Path(self.risk_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(log_path, index=False)
        print(f"\n📝 Risk rejection log saved: {log_path}")
        print(f"   Total rejections: {len(df)}")
        print(f"   Unique reasons: {df['Reason'].nunique()}")
