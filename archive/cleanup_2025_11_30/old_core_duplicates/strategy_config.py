"""
StrategyConfig - Flexible multi-strategy configuration system.

Allows configuring multiple signal generators with different assets and capital allocations.
Supports both single-asset and multi-asset signals with dynamic allocation.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from signals.base import SignalModel


class StrategyAllocation:
    """
    Represents a single strategy with its configuration.
    
    A strategy consists of:
    - Signal generator (e.g., MomentumSignalV2)
    - Assets to trade
    - Capital allocation (fixed or dynamic)
    - Strategy-specific parameters
    """
    
    def __init__(
        self,
        name: str,
        signal_generator: SignalModel,
        assets: List[str],
        capital_allocation: float,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize strategy allocation.
        
        Args:
            name: Strategy identifier (e.g., "Fast_Momentum")
            signal_generator: Instantiated signal generator
            assets: List of tickers this strategy trades
            capital_allocation: Capital allocated to this strategy ($ or fraction)
            params: Additional strategy-specific parameters
        """
        self.name = name
        self.signal_generator = signal_generator
        self.assets = assets
        self.capital_allocation = capital_allocation
        self.params = params or {}
    
    def generate_signals(self, prices: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate signals for this strategy's assets.
        
        Args:
            prices: Dict of {ticker: price_df}
            
        Returns:
            Dict of {ticker: signal_df} for this strategy's assets only
        """
        signals = {}
        
        # Generate signals for each asset individually
        for ticker in self.assets:
            if ticker in prices:
                # Call signal generator for single asset
                signal_df = self.signal_generator.generate(prices[ticker])
                signals[ticker] = signal_df
            else:
                raise ValueError(f"Asset '{ticker}' not found in prices data")
        
        return signals
    
    def __repr__(self):
        return (f"StrategyAllocation(name='{self.name}', "
                f"assets={self.assets}, "
                f"capital=${self.capital_allocation:,.0f})")


class StrategyConfig:
    """
    Multi-strategy configuration builder.
    
    Allows flexible configuration of multiple strategies, each with:
    - Different signal generators
    - Different assets
    - Different capital allocations
    - Different parameters
    
    Example:
        config = (StrategyConfig(total_capital=100000)
                  .add_strategy("Fast_Mom", MomentumSignalV2(60), ['ES', 'NQ'], 60000)
                  .add_strategy("Slow_Mom", MomentumSignalV2(120), ['GC'], 40000)
                  .build())
        
        signals = config.generate_all_signals(prices)
        allocations = config.get_allocations()
    """
    
    def __init__(self, total_capital: float = 100000):
        """
        Initialize strategy config builder.
        
        Args:
            total_capital: Total portfolio capital to allocate
        """
        self.total_capital = total_capital
        self.strategies: List[StrategyAllocation] = []
        self._built = False
    
    def add_strategy(
        self,
        name: str,
        signal_generator: SignalModel,
        assets: List[str],
        capital: Optional[float] = None,
        capital_pct: Optional[float] = None,
        **kwargs
    ) -> 'StrategyConfig':
        """
        Add a strategy to the configuration.
        
        Args:
            name: Strategy identifier
            signal_generator: Instantiated signal generator (e.g., MomentumSignalV2(60))
            assets: List of tickers (e.g., ['ES', 'NQ'])
            capital: Fixed capital allocation in dollars
            capital_pct: Capital allocation as percentage of total (e.g., 0.6 for 60%)
            **kwargs: Additional strategy parameters
            
        Returns:
            Self for chaining
            
        Example:
            config.add_strategy(
                "Fast_Momentum",
                MomentumSignalV2(lookback=60, sma_filter=150),
                ['ES', 'NQ'],
                capital=60000
            )
        """
        if self._built:
            raise ValueError("Cannot add strategies after build() has been called")
        
        # Determine capital allocation
        if capital is not None:
            allocation = capital
        elif capital_pct is not None:
            allocation = self.total_capital * capital_pct
        else:
            raise ValueError("Must specify either 'capital' or 'capital_pct'")
        
        # Create strategy allocation
        strategy = StrategyAllocation(
            name=name,
            signal_generator=signal_generator,
            assets=assets,
            capital_allocation=allocation,
            params=kwargs
        )
        
        self.strategies.append(strategy)
        return self
    
    def build(self) -> 'StrategyConfig':
        """
        Finalize configuration and validate.
        
        Returns:
            Self with validation complete
        """
        if self._built:
            return self
        
        # Validate total allocation
        total_allocated = sum(s.capital_allocation for s in self.strategies)
        
        if abs(total_allocated - self.total_capital) > 0.01:
            print(f"⚠️  Warning: Total allocated ${total_allocated:,.0f} "
                  f"!= Total capital ${self.total_capital:,.0f}")
            print(f"   Difference: ${total_allocated - self.total_capital:,.0f}")
        
        # Check for asset overlaps
        all_assets = []
        for strategy in self.strategies:
            for asset in strategy.assets:
                all_assets.append((strategy.name, asset))
        
        # Find duplicates
        from collections import Counter
        asset_counts = Counter([asset for _, asset in all_assets])
        overlaps = {asset: count for asset, count in asset_counts.items() if count > 1}
        
        if overlaps:
            print(f"⚠️  Warning: Assets traded by multiple strategies:")
            for asset, count in overlaps.items():
                strategies_with_asset = [name for name, a in all_assets if a == asset]
                print(f"   {asset}: {strategies_with_asset}")
            print(f"   This means signals will be combined/overlapped")
        
        self._built = True
        return self
    
    def generate_all_signals(self, prices: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Generate signals for all strategies.
        
        Args:
            prices: Dict of {ticker: price_df}
            
        Returns:
            Dict of {strategy_name: {ticker: signal_df}}
            
        Example:
            signals = config.generate_all_signals(prices)
            # signals = {
            #     'Fast_Momentum': {'ES': signal_df, 'NQ': signal_df},
            #     'Slow_Momentum': {'GC': signal_df}
            # }
        """
        if not self._built:
            raise ValueError("Must call build() before generating signals")
        
        all_signals = {}
        
        for strategy in self.strategies:
            signals = strategy.generate_signals(prices)
            all_signals[strategy.name] = signals
        
        return all_signals
    
    def get_allocations(self) -> Dict[str, float]:
        """
        Get capital allocation for each strategy.
        
        Returns:
            Dict of {strategy_name: capital_amount}
        """
        return {s.name: s.capital_allocation for s in self.strategies}
    
    def get_strategy_assets(self) -> Dict[str, List[str]]:
        """
        Get assets for each strategy.
        
        Returns:
            Dict of {strategy_name: [assets]}
        """
        return {s.name: s.assets for s in self.strategies}
    
    def get_all_assets(self) -> List[str]:
        """
        Get all unique assets across all strategies.
        
        Returns:
            List of unique asset tickers
        """
        all_assets = set()
        for strategy in self.strategies:
            all_assets.update(strategy.assets)
        return sorted(list(all_assets))
    
    def summary(self) -> str:
        """
        Get human-readable summary of configuration.
        
        Returns:
            Formatted summary string
        """
        lines = [
            "="*70,
            "STRATEGY CONFIGURATION",
            "="*70,
            f"Total Capital: ${self.total_capital:,.2f}",
            f"Number of Strategies: {len(self.strategies)}",
            f"Unique Assets: {len(self.get_all_assets())} {self.get_all_assets()}",
            "",
            "Strategy Allocations:",
            "-"*70
        ]
        
        for strategy in self.strategies:
            pct = strategy.capital_allocation / self.total_capital * 100
            lines.append(f"  {strategy.name:20} ${strategy.capital_allocation:>10,.0f} ({pct:>5.1f}%)")
            lines.append(f"     Assets: {strategy.assets}")
            lines.append(f"     Signal: {strategy.signal_generator.__class__.__name__}")
            lines.append("")
        
        lines.append("="*70)
        return "\n".join(lines)
    
    def __repr__(self):
        return f"StrategyConfig(total_capital=${self.total_capital:,.0f}, strategies={len(self.strategies)})"
