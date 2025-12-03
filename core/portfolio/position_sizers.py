"""
Position Sizing Strategies

Abstract base class and concrete implementations for calculating position sizes.
Separates sizing logic from risk management enforcement.

Available Sizers:
- FixedFractionalSizer: Fixed % of capital per position
- KellySizer: Kelly Criterion based on win rate and avg win/loss
- ATRSizer: Volatility-based sizing using Average True Range
- VolatilityScaledSizer: Inverse volatility weighting
- RiskParitySizer: Equal risk contribution across positions
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
import numpy as np
import pandas as pd


class PositionSizer(ABC):
    """Abstract base class for position sizing strategies."""
    
    @abstractmethod
    def calculate_size(
        self,
        ticker: str,
        signal: float,
        current_price: float,
        portfolio_value: float,
        **kwargs
    ) -> float:
        """
        Calculate position size in number of shares.
        
        Args:
            ticker: Asset ticker
            signal: Signal strength (-1 to 1, where 1 = full long, -1 = full short)
            current_price: Current price per share
            portfolio_value: Current portfolio value
            **kwargs: Additional parameters (volatility, stop_loss, etc.)
            
        Returns:
            Number of shares (integer)
        """
        pass
    
    def _apply_signal_scaling(self, base_size: float, signal: float) -> float:
        """Scale position size by signal strength."""
        return base_size * abs(signal)
    
    def _round_to_shares(self, size: float, min_value: float = 100.0, price: float = 1.0) -> int:
        """Round to whole shares and check minimum trade value."""
        shares = int(size)
        if shares * price < min_value:
            return 0
        return shares


class FixedFractionalSizer(PositionSizer):
    """
    Fixed fractional position sizing.
    
    Allocates a fixed percentage of portfolio to each position.
    Uses risk-based sizing if stop_loss is provided.
    
    Example:
        sizer = FixedFractionalSizer(max_position_pct=0.20, risk_per_trade=0.02)
        shares = sizer.calculate_size('ES', signal=1.0, current_price=4500, 
                                      portfolio_value=100000, stop_loss_pct=0.10)
    """
    
    def __init__(
        self,
        max_position_pct: float = 0.20,
        risk_per_trade: float = 0.02,
        min_trade_value: float = 100.0
    ):
        """
        Initialize fixed fractional sizer.
        
        Args:
            max_position_pct: Maximum position size as % of portfolio (e.g., 0.20 = 20%)
            risk_per_trade: Risk per trade as % of portfolio (e.g., 0.02 = 2%)
            min_trade_value: Minimum dollar value for a trade
        """
        self.max_position_pct = max_position_pct
        self.risk_per_trade = risk_per_trade
        self.min_trade_value = min_trade_value
    
    def calculate_size(
        self,
        ticker: str,
        signal: float,
        current_price: float,
        portfolio_value: float,
        stop_loss_pct: Optional[float] = None,
        **kwargs
    ) -> float:
        """Calculate position size using fixed fractional method."""
        if portfolio_value <= 0 or current_price <= 0:
            return 0
        
        # Method 1: Fixed percentage of portfolio
        max_position_value = portfolio_value * self.max_position_pct
        shares_from_size_limit = max_position_value / current_price
        
        # Method 2: Risk-based sizing (if stop loss provided)
        if stop_loss_pct and stop_loss_pct > 0:
            risk_amount = portfolio_value * self.risk_per_trade
            loss_per_share = current_price * stop_loss_pct
            
            if loss_per_share > 0:
                shares_from_risk = risk_amount / loss_per_share
                shares = min(shares_from_size_limit, shares_from_risk)
            else:
                shares = shares_from_size_limit
        else:
            shares = shares_from_size_limit
        
        # Scale by signal strength
        shares = self._apply_signal_scaling(shares, signal)
        
        # Round to whole shares
        return self._round_to_shares(shares, self.min_trade_value, current_price)


class KellySizer(PositionSizer):
    """
    Kelly Criterion position sizing.
    
    Calculates optimal position size based on win rate and average win/loss ratio.
    Uses half-Kelly by default for safety.
    
    Formula: f* = (p*W - (1-p)) / W
    where p = win rate, W = avg_win / avg_loss
    
    Example:
        sizer = KellySizer(max_position_pct=0.30, kelly_fraction=0.5)
        shares = sizer.calculate_size('ES', signal=1.0, current_price=4500,
                                      portfolio_value=100000, win_rate=0.55,
                                      avg_win=0.03, avg_loss=0.02)
    """
    
    def __init__(
        self,
        max_position_pct: float = 0.30,
        kelly_fraction: float = 0.5,
        min_trade_value: float = 100.0
    ):
        """
        Initialize Kelly sizer.
        
        Args:
            max_position_pct: Maximum position size as % of portfolio
            kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly, common practice)
            min_trade_value: Minimum dollar value for a trade
        """
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        self.min_trade_value = min_trade_value
    
    def calculate_size(
        self,
        ticker: str,
        signal: float,
        current_price: float,
        portfolio_value: float,
        win_rate: float = 0.50,
        avg_win: float = 0.02,
        avg_loss: float = 0.02,
        **kwargs
    ) -> float:
        """Calculate position size using Kelly Criterion."""
        if portfolio_value <= 0 or current_price <= 0:
            return 0
        
        # Calculate Kelly percentage
        if avg_loss > 0:
            win_loss_ratio = avg_win / avg_loss
            kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
            kelly_pct = max(0, kelly_pct * self.kelly_fraction)
        else:
            kelly_pct = 0.0
        
        # Cap at maximum position size
        position_pct = min(kelly_pct, self.max_position_pct)
        
        # Calculate shares
        position_value = portfolio_value * position_pct
        shares = position_value / current_price
        
        # Scale by signal strength
        shares = self._apply_signal_scaling(shares, signal)
        
        # Round to whole shares
        return self._round_to_shares(shares, self.min_trade_value, current_price)


class ATRSizer(PositionSizer):
    """
    ATR-based volatility position sizing.
    
    Sizes positions based on Average True Range (ATR) to normalize risk
    across assets with different volatility profiles.
    
    Example:
        sizer = ATRSizer(risk_per_trade=0.02, atr_multiplier=2.0)
        shares = sizer.calculate_size('ES', signal=1.0, current_price=4500,
                                      portfolio_value=100000, atr=50.0)
    """
    
    def __init__(
        self,
        risk_per_trade: float = 0.02,
        atr_multiplier: float = 2.0,
        max_position_pct: float = 0.25,
        min_trade_value: float = 100.0
    ):
        """
        Initialize ATR sizer.
        
        Args:
            risk_per_trade: Risk per trade as % of portfolio
            atr_multiplier: Multiplier for ATR stop distance (e.g., 2.0 = 2x ATR)
            max_position_pct: Maximum position size as % of portfolio
            min_trade_value: Minimum dollar value for a trade
        """
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.max_position_pct = max_position_pct
        self.min_trade_value = min_trade_value
    
    def calculate_size(
        self,
        ticker: str,
        signal: float,
        current_price: float,
        portfolio_value: float,
        atr: Optional[float] = None,
        **kwargs
    ) -> float:
        """Calculate position size using ATR-based method."""
        if portfolio_value <= 0 or current_price <= 0:
            return 0
        
        # If no ATR provided, fall back to fixed fractional
        if atr is None or atr <= 0:
            position_value = portfolio_value * self.max_position_pct
            shares = position_value / current_price
        else:
            # Calculate risk-adjusted position size
            risk_amount = portfolio_value * self.risk_per_trade
            stop_distance = atr * self.atr_multiplier
            
            if stop_distance > 0:
                shares = risk_amount / stop_distance
                
                # Cap at maximum position size
                max_shares = (portfolio_value * self.max_position_pct) / current_price
                shares = min(shares, max_shares)
            else:
                shares = 0
        
        # Scale by signal strength
        shares = self._apply_signal_scaling(shares, signal)
        
        # Round to whole shares
        return self._round_to_shares(shares, self.min_trade_value, current_price)


class VolatilityScaledSizer(PositionSizer):
    """
    Volatility-scaled position sizing (inverse volatility weighting).
    
    Allocates more capital to low-volatility assets and less to high-volatility assets
    to equalize risk contribution across positions.
    
    Example:
        sizer = VolatilityScaledSizer(target_volatility=0.15, max_position_pct=0.30)
        shares = sizer.calculate_size('ES', signal=1.0, current_price=4500,
                                      portfolio_value=100000, volatility=0.20)
    """
    
    def __init__(
        self,
        target_volatility: float = 0.15,
        max_position_pct: float = 0.30,
        min_position_pct: float = 0.05,
        min_trade_value: float = 100.0
    ):
        """
        Initialize volatility-scaled sizer.
        
        Args:
            target_volatility: Target portfolio volatility (annualized)
            max_position_pct: Maximum position size as % of portfolio
            min_position_pct: Minimum position size as % of portfolio
            min_trade_value: Minimum dollar value for a trade
        """
        self.target_volatility = target_volatility
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.min_trade_value = min_trade_value
    
    def calculate_size(
        self,
        ticker: str,
        signal: float,
        current_price: float,
        portfolio_value: float,
        volatility: Optional[float] = None,
        **kwargs
    ) -> float:
        """Calculate position size using inverse volatility weighting."""
        if portfolio_value <= 0 or current_price <= 0:
            return 0
        
        # If no volatility provided, fall back to fixed fractional
        if volatility is None or volatility <= 0:
            position_pct = self.max_position_pct
        else:
            # Scale position inversely to volatility
            position_pct = self.target_volatility / volatility
            
            # Clamp to min/max bounds
            position_pct = max(self.min_position_pct, min(position_pct, self.max_position_pct))
        
        # Calculate shares
        position_value = portfolio_value * position_pct
        shares = position_value / current_price
        
        # Scale by signal strength
        shares = self._apply_signal_scaling(shares, signal)
        
        # Round to whole shares
        return self._round_to_shares(shares, self.min_trade_value, current_price)


class RiskParitySizer(PositionSizer):
    """
    Risk parity position sizing.
    
    Equalizes risk contribution across positions using volatility and correlation.
    Requires correlation matrix for multi-asset portfolios.
    
    Example:
        sizer = RiskParitySizer(max_position_pct=0.25)
        shares = sizer.calculate_size('ES', signal=1.0, current_price=4500,
                                      portfolio_value=100000, volatility=0.20,
                                      correlation_matrix=corr_matrix, current_positions=positions)
    """
    
    def __init__(
        self,
        max_position_pct: float = 0.25,
        min_trade_value: float = 100.0
    ):
        """
        Initialize risk parity sizer.
        
        Args:
            max_position_pct: Maximum position size as % of portfolio
            min_trade_value: Minimum dollar value for a trade
        """
        self.max_position_pct = max_position_pct
        self.min_trade_value = min_trade_value
    
    def calculate_size(
        self,
        ticker: str,
        signal: float,
        current_price: float,
        portfolio_value: float,
        volatility: Optional[float] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        current_positions: Optional[Dict] = None,
        **kwargs
    ) -> float:
        """Calculate position size using risk parity method."""
        if portfolio_value <= 0 or current_price <= 0:
            return 0
        
        # Simplified risk parity: inverse volatility if no correlation data
        if volatility is None or volatility <= 0:
            position_pct = self.max_position_pct
        else:
            # For single asset or without correlation, use inverse volatility
            # In full implementation, would solve for equal risk contribution
            position_pct = (1.0 / volatility) if volatility > 0 else self.max_position_pct
            
            # Normalize if we have current positions
            if current_positions:
                total_inv_vol = sum(
                    1.0 / pos.get('volatility', 0.15) 
                    for pos in current_positions.values() 
                    if pos.get('volatility', 0.15) > 0
                )
                total_inv_vol += (1.0 / volatility)
                position_pct = (1.0 / volatility) / total_inv_vol
            
            # Cap at maximum
            position_pct = min(position_pct, self.max_position_pct)
        
        # Calculate shares
        position_value = portfolio_value * position_pct
        shares = position_value / current_price
        
        # Scale by signal strength
        shares = self._apply_signal_scaling(shares, signal)
        
        # Round to whole shares
        return self._round_to_shares(shares, self.min_trade_value, current_price)


class FuturesContractSizer(PositionSizer):
    """
    Futures contract position sizing with integer contracts.
    
    Ensures position sizes are in whole contracts based on contract multiplier.
    Uses fixed fractional sizing but rounds to integer contracts.
    
    Example:
        # ES futures with 50x multiplier
        sizer = FuturesContractSizer(
            contract_multipliers={'ES': 50, 'CL': 1000, 'GC': 100},
            max_position_pct=0.25,
            risk_per_trade=0.02
        )
        contracts = sizer.calculate_size('ES', signal=1.0, current_price=4500, 
                                        portfolio_value=1000000)
    """
    
    def __init__(
        self,
        contract_multipliers: Dict[str, float],
        max_position_pct: float = 0.25,
        risk_per_trade: float = 0.02,
        min_contracts: int = 1
    ):
        """
        Initialize futures contract sizer.
        
        Args:
            contract_multipliers: Dict of ticker -> contract multiplier (e.g., {'ES': 50})
            max_position_pct: Maximum position size as % of portfolio
            risk_per_trade: Risk per trade as % of portfolio
            min_contracts: Minimum contracts per trade (default 1)
        """
        self.contract_multipliers = contract_multipliers
        self.max_position_pct = max_position_pct
        self.risk_per_trade = risk_per_trade
        self.min_contracts = min_contracts
    
    def calculate_size(
        self,
        ticker: str,
        signal: float,
        current_price: float,
        portfolio_value: float,
        **kwargs
    ) -> float:
        """Calculate position size in integer contracts."""
        if portfolio_value <= 0 or current_price <= 0 or abs(signal) < 0.01:
            return 0
        
        # Get contract multiplier
        multiplier = self.contract_multipliers.get(ticker, 1)
        
        # Calculate max allocation
        max_allocation = portfolio_value * self.max_position_pct
        
        # Calculate notional value per contract
        notional_per_contract = current_price * multiplier
        
        if notional_per_contract <= 0:
            return 0
        
        # Calculate fractional contracts
        contracts_float = max_allocation / notional_per_contract
        
        # Scale by signal strength
        contracts_float = contracts_float * abs(signal)
        
        # Round down to integer contracts
        contracts_int = int(np.floor(contracts_float))
        
        # Check minimum contracts
        if contracts_int < self.min_contracts:
            return 0
        
        # Return as "shares" (actually contracts, but portfolio manager treats as shares)
        return float(contracts_int)


# Convenience factory function
def create_position_sizer(
    method: str = 'fixed_fractional',
    **kwargs
) -> PositionSizer:
    """
    Factory function to create position sizers.
    
    Args:
        method: Sizing method ('fixed_fractional', 'kelly', 'atr', 'volatility_scaled', 
                               'risk_parity', 'futures_contract')
        **kwargs: Parameters for the chosen sizer
        
    Returns:
        PositionSizer instance
        
    Example:
        sizer = create_position_sizer('kelly', kelly_fraction=0.5, max_position_pct=0.30)
        sizer = create_position_sizer('futures_contract', 
                                     contract_multipliers={'ES': 50, 'CL': 1000})
    """
    sizers = {
        'fixed_fractional': FixedFractionalSizer,
        'kelly': KellySizer,
        'atr': ATRSizer,
        'volatility_scaled': VolatilityScaledSizer,
        'risk_parity': RiskParitySizer,
        'futures_contract': FuturesContractSizer
    }
    
    if method not in sizers:
        raise ValueError(f"Unknown sizing method: {method}. Choose from {list(sizers.keys())}")
    
    return sizers[method](**kwargs)
