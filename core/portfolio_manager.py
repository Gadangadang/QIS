"""
Multi-Asset Portfolio Manager
Manages capital allocation across multiple assets with drift-based rebalancing.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management."""
    initial_capital: float = 100000.0
    target_weights: Optional[Dict[str, float]] = None  # {'ES': 0.5, 'GC': 0.5}
    rebalance_threshold: float = 0.05  # Rebalance when drift > 5%
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly', or 'never'
    transaction_cost_bps: float = 3.0  # Transaction costs in basis points
    

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
        
        If 2 signals active: 50% each
        If 1 signal active: 100% to that signal
        If 0 signals active: 0% each
        
        Returns:
            {ticker: target_weight} where weights sum to 1.0 for active signals
        """
        active_signals = self.get_active_signals(signals)
        n_active = len(active_signals)
        
        if n_active == 0:
            return {ticker: 0.0 for ticker in signals}
        
        # Equal weight among active signals
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


def run_multi_asset_backtest(
    signals_dict: Dict[str, pd.DataFrame],
    prices_dict: Dict[str, pd.DataFrame],
    config: PortfolioConfig
) -> Tuple[PortfolioManager, pd.DataFrame, pd.DataFrame]:
    """
    Run a multi-asset backtest with portfolio management.
    
    Args:
        signals_dict: {ticker: df_with_Signal_column}
        prices_dict: {ticker: df_with_OHLC}
        config: PortfolioConfig
        
    Returns:
        (portfolio_manager, equity_curve_df, trades_df)
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
    
    # Initialize on first date
    first_date = all_dates[0]
    init_prices = {ticker: df.loc[first_date, 'Close'] 
                   for ticker, df in prices_dict.items()}
    init_signals = {ticker: df.loc[first_date, 'Signal'] 
                    for ticker, df in signals_dict.items()}
    
    pm.initialize_positions(init_prices, init_signals)
    pm.equity_curve.append(pm.get_portfolio_state(first_date))
    
    # Run through all dates
    for date in all_dates[1:]:
        # Get current prices and signals
        current_prices = {ticker: df.loc[date, 'Close'] 
                         for ticker, df in prices_dict.items()}
        current_signals = {ticker: df.loc[date, 'Signal'] 
                          for ticker, df in signals_dict.items()}
        
        # Update position values with current prices
        pm.update_positions(current_prices)
        
        # Check if signals changed (entries/exits)
        pm.update_signals(current_signals, current_prices, date)
        
        # Update position values again after signal changes
        pm.update_positions(current_prices)
        
        # Check if rebalancing needed (only among active positions)
        if pm.check_rebalance_needed(current_signals):
            pm.rebalance(current_prices, current_signals, date)
            pm.update_positions(current_prices)  # Update after rebalance
        
        # Record portfolio state
        pm.equity_curve.append(pm.get_portfolio_state(date))
    
    return pm, pm.get_equity_curve(), pm.get_trades_df()


if __name__ == "__main__":
    # Test the portfolio manager
    from core.multi_asset_loader import load_assets
    from core.multi_asset_signal import UnifiedMomentumSignal
    
    print("Testing PortfolioManager...")
    
    # Load data
    prices = load_assets(['ES', 'GC'], start_date='2020-01-01')
    
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
