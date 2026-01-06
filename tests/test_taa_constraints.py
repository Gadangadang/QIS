"""
Tests for TAA portfolio constraints module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from core.taa.constraints import (
    PositionConstraints,
    TrackingErrorConstraint,
    TransactionCostModel,
    TurnoverConstraint,
    OptimizationConstraints,
    load_constraints_from_config
)


class TestPositionConstraints:
    """Test PositionConstraints dataclass."""
    
    def test_default_values(self):
        """Test default constraint values."""
        constraints = PositionConstraints()
        assert constraints.long_only is True
        assert constraints.max_position == 0.35  # Default from actual implementation
        assert constraints.min_position == 0.0
        assert constraints.allow_cash is True
        assert constraints.max_cash == 0.20
    
    def test_custom_values(self):
        """Test custom constraint values."""
        constraints = PositionConstraints(
            long_only=False,
            max_position=0.30,
            min_position=0.05
        )
        assert constraints.long_only is False
        assert constraints.max_position == 0.30
        assert constraints.min_position == 0.05
    
    def test_validate_valid_weights(self):
        """Test validation passes for valid weights."""
        constraints = PositionConstraints(max_position=0.25, min_position=0.0)
        # validate() takes no arguments - it validates the constraints themselves
        constraints.validate()  # Should not raise
    
    def test_validate_exceeds_max(self):
        """Test validation fails when max_position > 1.0."""
        with pytest.raises(ValueError, match="cannot exceed 1.0"):
            constraints = PositionConstraints(max_position=1.5)
            constraints.validate()
    
    def test_validate_below_min(self):
        """Test validation fails when min > max."""
        with pytest.raises(ValueError, match="min_position must be <= max_position"):
            constraints = PositionConstraints(min_position=0.30, max_position=0.20)
            constraints.validate()
    
    def test_validate_negative_weight_long_only(self):
        """Test validation fails for negative min_position when long_only=True."""
        with pytest.raises(ValueError, match="must be >= 0 for long-only"):
            constraints = PositionConstraints(long_only=True, min_position=-0.10)
            constraints.validate()


class TestTrackingErrorConstraint:
    """Test TrackingErrorConstraint dataclass."""
    
    def test_default_values(self):
        """Test default tracking error values."""
        constraint = TrackingErrorConstraint()
        assert constraint.enabled is True
        assert constraint.max_te == 0.08
        assert constraint.benchmark_weights is None  # Default is None
    
    def test_validate_below_threshold(self):
        """Test validation passes when benchmark weights provided."""
        constraint = TrackingErrorConstraint(
            max_te=0.10,
            benchmark_weights={'SPY': 0.5, 'XLE': 0.5}
        )
        # Should not raise
        constraint.validate()
    
    def test_validate_exceeds_threshold(self):
        """Test validation fails when benchmark weights not provided."""
        constraint = TrackingErrorConstraint(max_te=0.08, enabled=True)
        
        with pytest.raises(ValueError, match="benchmark_weights required"):
            constraint.validate()
    
    
    def test_benchmark_weights_sum_to_one(self):
        """Test benchmark weights must sum to approximately 1.0."""
        with pytest.raises(ValueError, match="sum to"):
            constraint = TrackingErrorConstraint(
                enabled=True,
                benchmark_weights={'SPY': 0.6, 'XLE': 0.5}  # Sums to 1.1
            )
            constraint.validate()


class TestTransactionCostModel:
    """Test TransactionCostModel dataclass."""
    
    def test_default_values(self):
        """Test default transaction cost values."""
        model = TransactionCostModel()
        assert model.commission_bps == 5
        assert model.slippage_bps == 3
        assert model.vix_adjustment_enabled is True
        assert model.vix_threshold == 30
        assert model.vix_multiplier == 1.5
    
    def test_calculate_cost_no_vix(self):
        """Test cost calculation without VIX adjustment."""
        model = TransactionCostModel(commission_bps=5, slippage_bps=3)
        cost = model.calculate_cost(turnover=0.50, current_vix=None)
        expected = 0.50 * (5 + 3) / 10000
        assert cost == pytest.approx(expected)
    
    def test_calculate_cost_vix_below_threshold(self):
        """Test cost calculation with VIX below threshold."""
        model = TransactionCostModel(
            commission_bps=5,
            slippage_bps=3,
            vix_threshold=30
        )
        cost = model.calculate_cost(turnover=0.50, current_vix=25)
        expected = 0.50 * 8 / 10000
        assert cost == pytest.approx(expected)
    
    def test_calculate_cost_vix_above_threshold(self):
        """Test cost calculation with VIX above threshold."""
        model = TransactionCostModel(
            commission_bps=5,
            slippage_bps=3,
            vix_threshold=30,
            vix_multiplier=1.5
        )
        cost = model.calculate_cost(turnover=0.50, current_vix=35)
        expected = 0.50 * 8 / 10000 * 1.5
        assert cost == pytest.approx(expected)
    
    def test_calculate_cost_vix_disabled(self):
        """Test cost calculation with VIX adjustment disabled."""
        model = TransactionCostModel(
            commission_bps=5,
            slippage_bps=3,
            vix_adjustment_enabled=False,
            vix_multiplier=1.5
        )
        cost = model.calculate_cost(turnover=0.50, current_vix=50)
        expected = 0.50 * 8 / 10000  # No multiplier applied
        assert cost == pytest.approx(expected)


class TestTurnoverConstraint:
    """Test TurnoverConstraint dataclass."""
    
    def test_default_values(self):
        """Test default turnover values."""
        constraint = TurnoverConstraint()
        assert constraint.enabled is True
        assert constraint.max_monthly == 0.50
        assert constraint.penalize is True
        assert constraint.penalty_lambda == 0.001
    
    def test_validate_below_threshold(self):
        """Test validation passes when turnover valid."""
        constraint = TurnoverConstraint(max_monthly=0.50)
        constraint.validate()  # Should not raise
    
    def test_validate_exceeds_threshold(self):
        """Test validation fails when max_monthly out of bounds."""
        with pytest.raises(ValueError, match="should be between 0 and 2.0"):
            constraint = TurnoverConstraint(max_monthly=2.5)
            constraint.validate()


class TestOptimizationConstraints:
    """Test OptimizationConstraints master class."""
    
    def test_default_initialization(self):
        """Test constraint initialization with all required args."""
        constraints = OptimizationConstraints(
            position=PositionConstraints(),
            tracking_error=TrackingErrorConstraint(benchmark_weights={'SPY': 1.0}),
            transaction_costs=TransactionCostModel(),
            turnover=TurnoverConstraint()
        )
        
        assert isinstance(constraints.position, PositionConstraints)
        assert isinstance(constraints.tracking_error, TrackingErrorConstraint)
        assert isinstance(constraints.transaction_costs, TransactionCostModel)
        assert isinstance(constraints.turnover, TurnoverConstraint)
        assert constraints.risk_aversion == 2.0
    
    def test_custom_initialization(self):
        """Test custom constraint initialization."""
        constraints = OptimizationConstraints(
            position=PositionConstraints(max_position=0.30),
            tracking_error=TrackingErrorConstraint(benchmark_weights={'SPY': 1.0}),
            transaction_costs=TransactionCostModel(),
            turnover=TurnoverConstraint(),
            risk_aversion=3.0
        )
        assert constraints.position.max_position == 0.30
        assert constraints.risk_aversion == 3.0
    
    def test_validate_all(self):
        """Test validate_all doesn't raise with valid settings."""
        constraints = OptimizationConstraints(
            position=PositionConstraints(),
            tracking_error=TrackingErrorConstraint(benchmark_weights={'SPY': 1.0}),
            transaction_costs=TransactionCostModel(),
            turnover=TurnoverConstraint()
        )
        constraints.validate_all()  # Should not raise


class TestLoadConstraintsFromConfig:
    """Test loading constraints from YAML config."""
    
    def test_load_from_existing_config(self):
        """Test loading from actual config file."""
        config_path = Path('/Users/Sakarias/QuantTrading/config/taa_constraints.yaml')
        
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        constraints = load_constraints_from_config(str(config_path))
        
        assert isinstance(constraints, OptimizationConstraints)
        assert constraints.position.long_only is True
        assert constraints.position.max_position == 0.33  # From actual config
        assert constraints.tracking_error.enabled is True
        assert constraints.turnover.max_monthly == 0.50
        assert constraints.risk_aversion == 2.0
    
    def test_load_invalid_path(self):
        """Test loading from non-existent path raises error."""
        with pytest.raises(FileNotFoundError):
            load_constraints_from_config('nonexistent.yaml')
