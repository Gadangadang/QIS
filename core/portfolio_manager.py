"""
Multi-Asset Portfolio Manager
Manages capital allocation across multiple assets with drift-based rebalancing.

Architecture:
- PortfolioManager: Top-level orchestrator for portfolio management
- BacktestResult: Lightweight container for backtest results and metrics
- Future modules: RiskManager, Reporter, StrategySelector (for walk-forward)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from core.risk_manager import RiskManager


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management."""
    initial_capital: float = 100000.0
    target_weights: Optional[Dict[str, float]] = None  # {'ES': 0.5, 'GC': 0.5}
    rebalance_threshold: float = 0.05  # Rebalance when drift > 5%
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly', or 'never'
    transaction_cost_bps: float = 3.0  # Transaction costs in basis points
    risk_manager: Optional['RiskManager'] = None  # Optional risk management
    rejection_policy: str = 'skip'  # 'skip' or 'scale_down' for rejected trades
    

class PortfolioManager:
    """
    Manages multi-asset portfolio with static allocation and drift rebalancing.
    
    Features:
    - Static target weights (e.g., 50% ES, 50% GC)
    - Automatic rebalancing when drift exceeds threshold
    - Per-asset and aggregate PnL tracking
    - Transaction cost accounting for rebalancing
    """
    
    def __init__(self, config: PortfolioConfig):
        """
        Initialize portfolio manager.
        
        Args:
            config: PortfolioConfig instance with allocation rules
        """
        self.config = config
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions = {}  # {ticker: {'shares': 0, 'value': 0, 'weight': 0}}
        self.trades = []  # List of all trades including rebalances
        self.equity_curve = []  # Daily portfolio value
        
    def initialize_positions(self, prices: Dict[str, float], signals: Dict[str, int]):
        """
        Initialize positions based on active signals.
        Allocates capital equally among active signals.
        
        Args:
            prices: {ticker: initial_price}
            signals: {ticker: signal} (1=long, 0=flat, -1=short)
        """
        # Get active signals and calculate equal weights
        target_weights = self.calculate_target_allocation(signals)
        
        # Allocate capital according to active signals
        for ticker, target_weight in target_weights.items():
            signal = signals.get(ticker, 0)
            price = prices[ticker]
            
            if signal != 0 and target_weight > 0:
                # Calculate dollar allocation
                target_value = self.config.initial_capital * target_weight
                shares = target_value / price
                
                self.positions[ticker] = {
                    'shares': shares,
                    'entry_price': price,
                    'current_price': price,
                    'value': target_value,
                    'weight': target_weight
                }
                
                self.cash -= target_value
            else:
                # No position
                self.positions[ticker] = {
                    'shares': 0,
                    'entry_price': price,
                    'current_price': price,
                    'value': 0,
                    'weight': 0
                }
    
    def update_positions(self, prices: Dict[str, float]):
        """
        Update position values based on current prices.
        
        Args:
            prices: {ticker: current_price}
        """
        for ticker, price in prices.items():
            if ticker in self.positions:
                pos = self.positions[ticker]
                pos['current_price'] = price
                pos['value'] = pos['shares'] * price
                
        # Calculate total portfolio value
        total_value = self.cash + sum(pos['value'] for pos in self.positions.values())
        self.portfolio_value = total_value
        
        # Update weights
        for ticker in self.positions:
            if total_value > 0:
                self.positions[ticker]['weight'] = self.positions[ticker]['value'] / total_value
            else:
                self.positions[ticker]['weight'] = 0
    
    def check_rebalance_needed(self, signals: Dict[str, int]) -> bool:
        """
        Check if rebalancing is needed based on drift threshold.
        
        Only checks drift among ACTIVE positions (where signal != 0).
        
        Args:
            signals: Current signals to determine which positions are active
            
        Returns:
            True if any active asset has drifted beyond threshold
        """
        active_signals = self.get_active_signals(signals)
        n_active = len(active_signals)
        
        if n_active <= 1:
            # No rebalancing needed with 0 or 1 active position
            return False
        
        # Target weight for each active position
        target_weight = 1.0 / n_active
        
        # Check drift for active positions only
        for ticker in active_signals:
            current_weight = self.positions.get(ticker, {}).get('weight', 0)
            
            # Calculate weight among active positions only
            total_active_value = sum(
                self.positions.get(t, {}).get('value', 0) 
                for t in active_signals
            )
            
            if total_active_value > 0:
                weight_among_active = self.positions.get(ticker, {}).get('value', 0) / total_active_value
                drift = abs(weight_among_active - target_weight)
                
                if drift > self.config.rebalance_threshold:
                    return True
        
        return False
    
    def rebalance(self, prices: Dict[str, float], signals: Dict[str, int], date: pd.Timestamp):
        """
        Rebalance portfolio back to equal weights among active positions.
        
        Only rebalances ACTIVE positions (where signal != 0).
        Maintains equal weight among active positions.
        
        Args:
            prices: {ticker: current_price}
            signals: {ticker: current_signal}
            date: Current date for trade recording
        """
        active_signals = self.get_active_signals(signals)
        n_active = len(active_signals)
        
        if n_active <= 1:
            return  # Nothing to rebalance
        
        # Calculate total value in active positions
        total_active_value = sum(
            self.positions.get(ticker, {}).get('value', 0) 
            for ticker in active_signals
        )
        
        # Target value for each active position (equal split)
        target_value_per_asset = total_active_value / n_active
        
        for ticker in active_signals:
            price = prices[ticker]
            current_pos = self.positions.get(ticker, {})
            current_shares = current_pos.get('shares', 0)
            current_value = current_pos.get('value', 0)
            
            # Calculate target shares
            target_shares = target_value_per_asset / price
            shares_to_trade = target_shares - current_shares
            
            if abs(shares_to_trade * price) > 10:  # Avoid tiny trades (<$10)
                trade_value = shares_to_trade * price
                tc = abs(trade_value) * (self.config.transaction_cost_bps / 10000)
                
                # Execute trade
                self.positions[ticker]['shares'] = target_shares
                self.positions[ticker]['value'] = target_value_per_asset
                self.cash -= (trade_value + tc)
                
                # Record trade
                self.trades.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Type': 'Rebalance',
                    'Shares': shares_to_trade,
                    'Price': price,
                    'Value': trade_value,
                    'TransactionCost': tc,
                    'Signal': signals[ticker],
                    'CurrentValue': current_value,
                    'TargetValue': target_value_per_asset
                })
    
    def get_active_signals(self, signals: Dict[str, int]) -> Dict[str, int]:
        """Get only non-zero signals (active positions)."""
        return {ticker: sig for ticker, sig in signals.items() if sig != 0}
    
    def calculate_target_allocation(self, signals: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate target weights based on active signals.
        
        If risk_manager is configured, uses risk-adjusted position sizing.
        Otherwise, uses equal weight allocation.
        
        Returns:
            {ticker: target_weight} where weights sum to <= 1.0
        """
        active_signals = self.get_active_signals(signals)
        n_active = len(active_signals)
        
        if n_active == 0:
            return {ticker: 0.0 for ticker in signals}
        
        # If risk manager is configured, use risk-adjusted sizing
        if self.config.risk_manager:
            target_weights = {}
            total_weight = 0.0
            
            for ticker in signals:
                signal = signals[ticker]
                
                if ticker in active_signals and signal != 0:
                    # Calculate volatility if we have returns
                    vol = None
                    if ticker in self.config.risk_manager.returns_history:
                        returns = pd.Series(list(self.config.risk_manager.returns_history[ticker]))
                        if len(returns) >= 20:
                            vol = self.config.risk_manager.calculate_volatility(ticker, returns)
                    
                    # Get current positions for validation
                    current_positions = {t: pos['shares'] for t, pos in self.positions.items() 
                                       if pos['shares'] != 0}
                    
                    # Calculate risk-adjusted position size
                    position_size = self.config.risk_manager.calculate_position_size(
                        ticker=ticker,
                        signal=float(signal),
                        capital=self.portfolio_value,
                        positions=current_positions,
                        volatility=vol
                    )
                    
                    # Validate the position size
                    is_valid, reason = self.config.risk_manager.validate_trade(
                        ticker=ticker,
                        size=position_size,
                        positions=current_positions,
                        portfolio_value=self.portfolio_value
                    )
                    
                    if is_valid:
                        target_weights[ticker] = position_size
                        total_weight += position_size
                    else:
                        # Handle rejection based on policy
                        if self.config.rejection_policy == 'scale_down':
                            # Scale down to max allowed
                            scaled_size = min(position_size, self.config.risk_manager.config.max_position_size)
                            target_weights[ticker] = scaled_size
                            total_weight += scaled_size
                        else:  # 'skip'
                            target_weights[ticker] = 0.0
                            # Log the rejection
                            self.config.risk_manager._log_violation(ticker, 'REJECTED', reason)
                else:
                    target_weights[ticker] = 0.0
            
            return target_weights
        else:
            # Default: Equal weight among active signals
            equal_weight = 1.0 / n_active
            
            target_weights = {}
            for ticker in signals:
                if ticker in active_signals:
                    target_weights[ticker] = equal_weight
                else:
                    target_weights[ticker] = 0.0
            
            return target_weights
    
    def update_signals(self, signals: Dict[str, int], prices: Dict[str, float], date: pd.Timestamp):
        """
        Update positions when signals change (entries/exits).
        
        Logic for Option B: Minimize disruption to existing positions
        - Use available cash first for new entries
        - Only rebalance existing positions if necessary
        - Never sell a winning position unless signal says exit
        
        Args:
            signals: {ticker: new_signal}
            prices: {ticker: current_price}
            date: Current date
        """
        # Track which signals changed
        signal_changes = {}
        for ticker, new_signal in signals.items():
            pos = self.positions.get(ticker, {})
            current_shares = pos.get('shares', 0)
            current_signal = np.sign(current_shares)
            
            if new_signal != current_signal:
                signal_changes[ticker] = {
                    'old': current_signal,
                    'new': new_signal,
                    'price': prices[ticker]
                }
        
        if not signal_changes:
            return  # No signal changes
        
        # Calculate target weights based on NEW signals
        target_weights = self.calculate_target_allocation(signals)
        
        # Process exits first (free up cash)
        for ticker, change in signal_changes.items():
            if change['new'] == 0:  # Exit signal
                price = change['price']
                pos = self.positions[ticker]
                shares_to_sell = pos['shares']
                trade_value = shares_to_sell * price
                tc = abs(trade_value) * (self.config.transaction_cost_bps / 10000)
                
                # Close position
                self.cash += (trade_value - tc)
                self.positions[ticker]['shares'] = 0
                self.positions[ticker]['value'] = 0
                
                # Record trade
                self.trades.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Type': 'Exit',
                    'Shares': -shares_to_sell,
                    'Price': price,
                    'Value': -trade_value,
                    'TransactionCost': tc,
                    'Signal': 0,
                    'PrevSignal': change['old']
                })
        
        # Process entries using available cash (minimize selling)
        for ticker, change in signal_changes.items():
            if change['new'] != 0 and change['old'] == 0:  # New entry
                price = change['price']
                target_weight = target_weights[ticker]
                
                # Ideal target value for this position
                ideal_value = self.portfolio_value * target_weight
                
                # How much can we afford with available cash?
                affordable_value = min(ideal_value, self.cash * 0.99)  # Keep 1% cash buffer
                
                if affordable_value > 100:  # Only trade if > $100
                    shares_to_buy = affordable_value / price
                    trade_value = shares_to_buy * price
                    tc = abs(trade_value) * (self.config.transaction_cost_bps / 10000)
                    
                    # Enter position with available cash
                    self.cash -= (trade_value + tc)
                    self.positions[ticker]['shares'] = shares_to_buy
                    self.positions[ticker]['entry_price'] = price
                    self.positions[ticker]['value'] = trade_value
                    
                    # Record trade
                    self.trades.append({
                        'Date': date,
                        'Ticker': ticker,
                        'Type': 'Entry',
                        'Shares': shares_to_buy,
                        'Price': price,
                        'Value': trade_value,
                        'TransactionCost': tc,
                        'Signal': change['new'],
                        'PrevSignal': 0,
                        'Note': 'Used available cash'
                    })
        
        # Process signal flips (e.g., long -> short)
        for ticker, change in signal_changes.items():
            if change['new'] != 0 and change['old'] != 0 and change['new'] != change['old']:
                price = change['price']
                target_weight = target_weights[ticker]
                target_value = self.portfolio_value * target_weight * change['new']
                target_shares = target_value / price
                
                pos = self.positions[ticker]
                shares_to_trade = target_shares - pos['shares']
                trade_value = shares_to_trade * price
                tc = abs(trade_value) * (self.config.transaction_cost_bps / 10000)
                
                # Flip position
                self.cash -= (trade_value + tc)
                self.positions[ticker]['shares'] = target_shares
                self.positions[ticker]['entry_price'] = price
                
                # Record trade
                self.trades.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Type': 'Flip',
                    'Shares': shares_to_trade,
                    'Price': price,
                    'Value': trade_value,
                    'TransactionCost': tc,
                    'Signal': change['new'],
                    'PrevSignal': change['old']
                })
    
    def get_portfolio_state(self, date: pd.Timestamp) -> Dict:
        """Get current portfolio state snapshot."""
        return {
            'Date': date,
            'TotalValue': self.portfolio_value,
            'Cash': self.cash,
            'Positions': {ticker: {
                'Shares': pos['shares'],
                'Price': pos['current_price'],
                'Value': pos['value'],
                'Weight': pos['weight']
            } for ticker, pos in self.positions.items()}
        }
    
    def get_trades_df(self) -> pd.DataFrame:
        """Return trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Return equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_curve)
    
    def calculate_metrics(self) -> Dict:
        """Calculate portfolio performance metrics."""
        equity_df = self.get_equity_curve()
        if equity_df.empty:
            return {}
        
        equity_df['Return'] = equity_df['TotalValue'].pct_change()
        
        total_return = (equity_df['TotalValue'].iloc[-1] / self.config.initial_capital) - 1
        
        # Annualized metrics
        n_days = len(equity_df)
        years = n_days / 252
        cagr = ((equity_df['TotalValue'].iloc[-1] / self.config.initial_capital) ** (1/years)) - 1 if years > 0 else 0
        
        # Volatility
        daily_vol = equity_df['Return'].std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe (assuming 0% risk-free rate)
        sharpe = (equity_df['Return'].mean() / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0
        
        # Drawdown
        running_max = equity_df['TotalValue'].expanding().max()
        drawdown = (equity_df['TotalValue'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        trades_df = self.get_trades_df()
        n_trades = len(trades_df)
        n_rebalances = len(trades_df[trades_df['Type'] == 'Rebalance']) if not trades_df.empty else 0
        total_tc = trades_df['TransactionCost'].sum() if not trades_df.empty else 0
        
        return {
            'Total Return': total_return,
            'CAGR': cagr,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Total Trades': n_trades,
            'Rebalances': n_rebalances,
            'Transaction Costs': total_tc,
            'TC as % of Capital': total_tc / self.config.initial_capital
        }


class BacktestResult:
    """
    Lightweight result container for backtest outputs.
    
    Provides a clean interface for accessing backtest results and metrics
    without requiring the full PortfolioManager state machine.
    
    This is useful for:
    - Walk-forward optimization (lightweight result objects)
    - Comparing multiple backtest runs
    - Generating reports without full portfolio state
    """
    
    def __init__(self, equity_curve: pd.DataFrame, trades: pd.DataFrame, config: PortfolioConfig,
                 risk_metrics: Optional[pd.DataFrame] = None, violations: Optional[pd.DataFrame] = None):
        """
        Initialize backtest result.
        
        Args:
            equity_curve: DataFrame with Date and TotalValue columns
            trades: DataFrame with trade details
            config: PortfolioConfig used for the backtest
            risk_metrics: Optional DataFrame with risk metrics history
            violations: Optional DataFrame with trade rejections/violations
        """
        self.config = config
        self._equity_curve = equity_curve
        self._trades = trades
        self.risk_metrics = risk_metrics
        self.violations = violations
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Return equity curve as DataFrame."""
        return self._equity_curve
    
    def get_trades_df(self) -> pd.DataFrame:
        """Return trades as DataFrame."""
        return self._trades
    
    def calculate_metrics(self) -> Dict:
        """Calculate portfolio performance metrics."""
        equity_df = self._equity_curve.copy()
        if equity_df.empty or 'TotalValue' not in equity_df.columns:
            return {}
        
        equity_df['Return'] = equity_df['TotalValue'].pct_change()
        
        total_return = (equity_df['TotalValue'].iloc[-1] / self.config.initial_capital) - 1
        
        # Annualized metrics
        n_days = len(equity_df)
        years = n_days / 252
        cagr = ((equity_df['TotalValue'].iloc[-1] / self.config.initial_capital) ** (1/years)) - 1 if years > 0 else 0
        
        # Volatility
        daily_vol = equity_df['Return'].std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe (assuming 0% risk-free rate)
        sharpe = (equity_df['Return'].mean() / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0
        
        # Drawdown
        running_max = equity_df['TotalValue'].expanding().max()
        drawdown = (equity_df['TotalValue'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        trades_df = self._trades
        n_trades = len(trades_df)
        n_rebalances = len(trades_df[trades_df['Type'] == 'Rebalance']) if not trades_df.empty and 'Type' in trades_df.columns else 0
        total_tc = trades_df['TransactionCost'].sum() if not trades_df.empty and 'TransactionCost' in trades_df.columns else 0
        
        return {
            'Total Return': total_return,
            'CAGR': cagr,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Total Trades': n_trades,
            'Rebalances': n_rebalances,
            'Transaction Costs': total_tc,
            'TC as % of Capital': total_tc / self.config.initial_capital
        }


def run_multi_asset_backtest(
    signals_dict: Dict[str, pd.DataFrame],
    prices_dict: Dict[str, pd.DataFrame],
    config: PortfolioConfig,
    return_pm: bool = False
) -> Tuple['PortfolioManager', pd.DataFrame, pd.DataFrame]:
    """
    Run a multi-asset backtest with portfolio management.
    
    This is the main entry point for backtesting. It orchestrates the backtest
    process and can return either the full PortfolioManager state or a lightweight
    BacktestResult object.
    
    Args:
        signals_dict: {ticker: df_with_Signal_column}
        prices_dict: {ticker: df_with_OHLC}
        config: PortfolioConfig
        return_pm: If True, return full PortfolioManager object
                   If False, return lightweight BacktestResult
        
    Returns:
        (backtest_result, equity_curve_df, trades_df)
        - backtest_result: PortfolioManager or BacktestResult object
        - equity_curve_df: DataFrame with portfolio equity over time
        - trades_df: DataFrame with all trades
    """
    pm = _run_backtest(signals_dict, prices_dict, config)
    equity_curve = pm.get_equity_curve()
    trades = pm.get_trades_df()
    
    # Extract risk metrics if risk manager was used
    risk_metrics = None
    violations = None
    if config.risk_manager:
        risk_metrics = config.risk_manager.get_metrics_dataframe()
        violations = config.risk_manager.get_violations_dataframe()
    
    if return_pm:
        return pm, equity_curve, trades
    else:
        # Return lightweight BacktestResult with risk data
        result = BacktestResult(equity_curve, trades, config, risk_metrics, violations)
        return result, equity_curve, trades


def _run_backtest(
    signals_dict: Dict[str, pd.DataFrame],
    prices_dict: Dict[str, pd.DataFrame],
    config: PortfolioConfig
) -> PortfolioManager:
    """
    Core backtest implementation.
    
    Runs the portfolio through historical data, handling:
    - Signal changes (entries/exits)
    - Position value updates
    - Drift-based rebalancing
    - Transaction cost tracking
    - Risk management (if configured)
    """
    # Initialize portfolio manager
    pm = PortfolioManager(config)
    
    # Get common dates from index
    all_dates = None
    for df in signals_dict.values():
        dates = set(df.index)
        if all_dates is None:
            all_dates = dates
        else:
            all_dates &= dates
    
    all_dates = sorted(all_dates)
    
    # Track previous prices for returns calculation (if using risk manager)
    prev_prices = {}
    
    # Initialize on first date
    first_date = all_dates[0]
    init_prices = {ticker: df.loc[first_date, 'Close'] 
                   for ticker, df in prices_dict.items()}
    init_signals = {ticker: df.loc[first_date, 'Signal'] 
                    for ticker, df in signals_dict.items()}
    
    pm.initialize_positions(init_prices, init_signals)
    pm.equity_curve.append(pm.get_portfolio_state(first_date))
    
    # Initialize previous prices
    prev_prices = init_prices.copy()
    
    # Initialize correlation matrix if risk manager exists
    if config.risk_manager:
        # Build initial returns dataframe for correlation
        # Use the first 60+ days of the backtest period for initialization
        returns_data = {}
        for ticker, df in prices_dict.items():
            # Try to get data before first_date (historical warm-up data)
            mask_before = df.index < first_date
            
            if mask_before.sum() >= 60:
                # We have historical data before backtest start - use it!
                hist_data = df.loc[mask_before, 'Close'].tail(60)
                returns_data[ticker] = hist_data.pct_change().dropna()
            else:
                # No historical data - use first 60 days of backtest period
                mask_first = (df.index >= first_date) & (df.index <= df.index[min(59, len(df)-1)])
                if mask_first.sum() >= 20:  # Need at least 20 days
                    hist_data = df.loc[mask_first, 'Close']
                    returns_data[ticker] = hist_data.pct_change().dropna()
        
        if returns_data and len(returns_data) >= 2:  # Need at least 2 assets
            returns_df = pd.DataFrame(returns_data).dropna()
            if len(returns_df) >= 10:  # Minimum 10 returns for correlation
                config.risk_manager.update_correlations(returns_df)
    
    # Track iteration for periodic correlation updates
    iteration_count = 0
    
    # Run through all dates
    for date in all_dates[1:]:
        iteration_count += 1
        
        # Get current prices and signals
        current_prices = {ticker: df.loc[date, 'Close'] 
                         for ticker, df in prices_dict.items()}
        current_signals = {ticker: df.loc[date, 'Signal'] 
                          for ticker, df in signals_dict.items()}
        
        # Update risk manager with daily returns
        if config.risk_manager:
            for ticker in current_prices:
                if ticker in prev_prices and prev_prices[ticker] > 0:
                    daily_return = (current_prices[ticker] / prev_prices[ticker]) - 1
                    config.risk_manager.update_returns(ticker, pd.Timestamp(date), daily_return)
            
            # Update correlations periodically (every 20 days) for better heatmap
            if iteration_count % 20 == 0 and len(config.risk_manager.returns_history) > 0:
                # Build returns dataframe from returns_history buffers
                returns_for_corr = {}
                for ticker, returns_deque in config.risk_manager.returns_history.items():
                    if len(returns_deque) >= 60:
                        returns_for_corr[ticker] = list(returns_deque)
                
                if len(returns_for_corr) >= 2:  # Need at least 2 assets for correlation
                    returns_df_update = pd.DataFrame(returns_for_corr)
                    config.risk_manager.update_correlations(returns_df_update)
        
        # Update position values with current prices
        pm.update_positions(current_prices)
        
        # Check for drawdown stop before trading
        if config.risk_manager:
            current_value = pm.portfolio_value
            equity_series = pd.Series([state['TotalValue'] for state in pm.equity_curve])
            peak = max(equity_series.max(), current_value) if len(equity_series) > 0 else current_value
            current_dd = min(0, (current_value - peak) / peak) if peak > 0 else 0  # Drawdown is always ≤ 0
            
            should_stop, reason = config.risk_manager.check_stop_conditions(
                current_drawdown=current_dd,
                equity_curve=equity_series
            )
            
            if should_stop:
                # Log violation and stop trading
                config.risk_manager._log_violation('PORTFOLIO', 'STOP', reason)
                # Update timestamp with actual date
                if config.risk_manager.violations_history:
                    config.risk_manager.violations_history[-1]['date'] = pd.Timestamp(date)
                    config.risk_manager.violations_history[-1]['timestamp'] = pd.Timestamp(date)
                # Record final state and exit
                pm.equity_curve.append(pm.get_portfolio_state(date))
                break
        
        # Check if signals changed (entries/exits)
        pm.update_signals(current_signals, current_prices, date)
        
        # Update position values again after signal changes
        pm.update_positions(current_prices)
        
        # Check if rebalancing needed (only among active positions)
        if pm.check_rebalance_needed(current_signals):
            pm.rebalance(current_prices, current_signals, date)
            pm.update_positions(current_prices)  # Update after rebalance
        
        # Collect risk metrics if risk manager is configured
        if config.risk_manager:
            current_value = pm.portfolio_value
            
            # Get current positions as shares dict
            current_positions = {ticker: pos['shares'] 
                               for ticker, pos in pm.positions.items() 
                               if pos['shares'] != 0}
            
            # Calculate drawdown
            equity_series = pd.Series([state['TotalValue'] for state in pm.equity_curve])
            peak = max(equity_series.max(), current_value) if len(equity_series) > 0 else current_value
            drawdown = min(0, (current_value - peak) / peak) if peak > 0 else 0  # Drawdown is always ≤ 0
            
            # Log metrics
            config.risk_manager.log_metrics(
                date=pd.Timestamp(date),
                positions=current_positions,
                prices=current_prices,
                portfolio_value=current_value,
                drawdown=drawdown
            )
        
        # Record portfolio state
        pm.equity_curve.append(pm.get_portfolio_state(date))
        
        # Update previous prices for next iteration
        prev_prices = current_prices.copy()
    
    return pm


if __name__ == "__main__":
    # Test the portfolio manager
    from core.multi_asset_loader import load_assets
    from core.multi_asset_signal import UnifiedMomentumSignal
    
    print("Testing PortfolioManager...")
    
    # Load data
    prices = load_assets(['ES', 'GC'], start_date='2015-01-01')
    
    # Generate signals
    signal_gen = UnifiedMomentumSignal(lookback=50, entry_z=2.0, exit_z=0.5, sma_period=200)
    signals = signal_gen.generate(prices)
    
    # Setup portfolio config
    config = PortfolioConfig(
        initial_capital=100000,
        target_weights={'ES': 0.5, 'GC': 0.5},
        rebalance_threshold=0.05,
        transaction_cost_bps=3.0
    )
    
    # Run backtest
    print("\nRunning backtest...")
    pm, equity_curve, trades = run_multi_asset_backtest(signals, prices, config)
    
    # Show results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    metrics = pm.calculate_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'Return' in key or 'CAGR' in key or 'Volatility' in key or 'Drawdown' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "="*60)
    print("TRADES SUMMARY")
    print("="*60)
    if not trades.empty:
        print(f"Total trades: {len(trades)}")
        print(f"\nBy type:")
        print(trades['Type'].value_counts())
        print(f"\nBy ticker:")
        print(trades['Ticker'].value_counts())
    else:
        print("No trades executed")
