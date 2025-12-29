"""
Portfolio Constraints for TAA Optimization.
Defines constraint classes used by the optimizer.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class PositionConstraints:
    """Constraints on individual position sizes."""
    
    long_only: bool = True
    min_position: float = 0.0
    max_position: float = 0.35
    allow_cash: bool = True
    max_cash: float = 0.20
    
    def validate(self):
        """Validate constraint parameters."""
        if self.min_position < 0 and self.long_only:
            raise ValueError("min_position must be >= 0 for long-only portfolios")
        if self.max_position > 1.0:
            raise ValueError("max_position cannot exceed 1.0")
        if self.min_position > self.max_position:
            raise ValueError("min_position must be <= max_position")


@dataclass
class TrackingErrorConstraint:
    """Tracking error constraint relative to benchmark."""
    
    enabled: bool = True
    max_te: float = 0.08  # 8% annual TE
    benchmark_weights: Optional[Dict[str, float]] = None
    
    def validate(self):
        """Validate constraint parameters."""
        if self.enabled and self.max_te <= 0:
            raise ValueError("max_te must be positive")
        if self.enabled and self.benchmark_weights is None:
            raise ValueError("benchmark_weights required when tracking error is enabled")
        
        if self.benchmark_weights:
            total = sum(self.benchmark_weights.values())
            if not np.isclose(total, 1.0, atol=0.01):
                raise ValueError(f"Benchmark weights sum to {total:.4f}, expected 1.0")


@dataclass
class TransactionCostModel:
    """Transaction cost parameters."""
    
    commission_bps: float = 5.0
    slippage_bps: float = 3.0
    vix_adjustment_enabled: bool = True
    vix_threshold: float = 30.0
    vix_multiplier: float = 1.5
    
    def calculate_cost(self, turnover: float, current_vix: Optional[float] = None) -> float:
        """
        Calculate total transaction cost.
        
        Args:
            turnover: Total turnover as fraction of portfolio (e.g., 0.5 for 50%)
            current_vix: Current VIX level (optional, for volatility adjustment)
            
        Returns:
            float: Total cost in basis points
        """
        base_cost = self.commission_bps + self.slippage_bps
        
        # Adjust for volatility
        if self.vix_adjustment_enabled and current_vix is not None:
            if current_vix > self.vix_threshold:
                base_cost *= self.vix_multiplier
        
        return base_cost * turnover / 10000  # Convert bps to fraction


@dataclass
class TurnoverConstraint:
    """Turnover limit constraint."""
    
    enabled: bool = True
    max_monthly: float = 0.50  # 50% max monthly turnover
    penalize: bool = True
    penalty_lambda: float = 0.001
    
    def validate(self):
        """Validate constraint parameters."""
        if self.max_monthly < 0 or self.max_monthly > 2.0:
            raise ValueError("max_monthly should be between 0 and 2.0")


@dataclass
class OptimizationConstraints:
    """
    Complete set of optimization constraints.
    
    Example:
        >>> constraints = OptimizationConstraints(
        ...     position=PositionConstraints(max_position=0.25),
        ...     tracking_error=TrackingErrorConstraint(max_te=0.08),
        ...     transaction_costs=TransactionCostModel(commission_bps=5),
        ...     turnover=TurnoverConstraint(max_monthly=0.5)
        ... )
    """
    
    position: PositionConstraints
    tracking_error: TrackingErrorConstraint
    transaction_costs: TransactionCostModel
    turnover: TurnoverConstraint
    risk_aversion: float = 2.0
    
    def validate_all(self):
        """Validate all constraints."""
        self.position.validate()
        self.tracking_error.validate()
        self.turnover.validate()
        
        if self.risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")


def load_constraints_from_config(config_path: str) -> OptimizationConstraints:
    """
    Load constraints from YAML config file.
    
    Args:
        config_path: Path to taa_constraints.yaml
        
    Returns:
        OptimizationConstraints: Parsed constraints object
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse position constraints
    portfolio = config['portfolio']
    position = PositionConstraints(
        long_only=portfolio['long_only'],
        min_position=portfolio['min_position'],
        max_position=portfolio['max_position'],
        allow_cash=portfolio.get('allow_cash', True),
        max_cash=portfolio.get('max_cash', 0.20)
    )
    
    # Parse tracking error
    te_config = config['tracking_error']
    tracking_error = TrackingErrorConstraint(
        enabled=te_config['enabled'],
        max_te=te_config['max_te'],
        benchmark_weights=te_config.get('benchmark_weights')
    )
    
    # Parse transaction costs
    tc_config = config['transaction_costs']
    transaction_costs = TransactionCostModel(
        commission_bps=tc_config['commission_bps'],
        slippage_bps=tc_config['slippage_bps'],
        vix_adjustment_enabled=tc_config['vix_adjustment']['enabled'],
        vix_threshold=tc_config['vix_adjustment']['threshold'],
        vix_multiplier=tc_config['vix_adjustment']['multiplier']
    )
    
    # Parse turnover
    to_config = config['turnover']
    turnover = TurnoverConstraint(
        enabled=True,
        max_monthly=to_config['max_monthly'],
        penalize=to_config['penalize'],
        penalty_lambda=to_config['penalty_lambda']
    )
    
    constraints = OptimizationConstraints(
        position=position,
        tracking_error=tracking_error,
        transaction_costs=transaction_costs,
        turnover=turnover,
        risk_aversion=config['risk_aversion']
    )
    
    constraints.validate_all()
    return constraints
