"""
Comprehensive unit tests for ExecutionEngine.
Tests for transaction costs, slippage, and order execution simulation.
"""

import pytest
from core.portfolio.execution_engine import ExecutionEngine, ExecutionConfig


class TestExecutionConfig:
    """Test suite for ExecutionConfig dataclass."""

    def test_default_config(self):
        """Test ExecutionConfig with default values."""
        config = ExecutionConfig()
        
        assert config.transaction_cost_bps == 3.0
        assert config.slippage_bps == 2.0
        assert config.min_trade_value == 100.0
        assert config.market_impact_factor == 0.0

    def test_custom_config(self):
        """Test ExecutionConfig with custom values."""
        config = ExecutionConfig(
            transaction_cost_bps=5.0,
            slippage_bps=3.0,
            min_trade_value=500.0,
            market_impact_factor=0.001
        )
        
        assert config.transaction_cost_bps == 5.0
        assert config.slippage_bps == 3.0
        assert config.min_trade_value == 500.0
        assert config.market_impact_factor == 0.001


class TestExecutionEngineBuy:
    """Test suite for ExecutionEngine.execute_buy method."""

    def test_execute_buy_basic(self):
        """Test basic buy execution with costs."""
        config = ExecutionConfig(transaction_cost_bps=3.0, slippage_bps=2.0)
        engine = ExecutionEngine(config)
        
        fill_price, cost = engine.execute_buy('AAPL', shares=100, market_price=150.0)
        
        # Slippage: 150 * 0.0002 = 0.03 per share
        # Fill price: 150 + 0.03 = 150.03
        expected_fill = 150.0 + (150.0 * 0.0002)
        assert pytest.approx(fill_price, rel=1e-6) == expected_fill
        
        # Transaction cost: 100 * 150.03 * 0.0003 = 4.509
        trade_value = 100 * fill_price
        expected_cost = trade_value * 0.0003
        assert pytest.approx(cost, rel=1e-6) == expected_cost

    def test_execute_buy_zero_costs(self):
        """Test buy execution with zero transaction costs."""
        config = ExecutionConfig(transaction_cost_bps=0.0, slippage_bps=0.0)
        engine = ExecutionEngine(config)
        
        fill_price, cost = engine.execute_buy('AAPL', shares=100, market_price=150.0)
        
        assert fill_price == 150.0  # No slippage
        assert cost == 0.0  # No transaction cost

    def test_execute_buy_high_costs(self):
        """Test buy execution with high transaction costs."""
        config = ExecutionConfig(transaction_cost_bps=50.0, slippage_bps=20.0)
        engine = ExecutionEngine(config)
        
        fill_price, cost = engine.execute_buy('AAPL', shares=100, market_price=100.0)
        
        # Slippage: 100 * 0.002 = 0.2 per share
        assert pytest.approx(fill_price, rel=1e-4) == 100.2
        
        # Transaction cost should be significant
        assert cost > 50.0

    def test_execute_buy_large_order(self):
        """Test buy execution with large order."""
        config = ExecutionConfig()
        engine = ExecutionEngine(config)
        
        fill_price, cost = engine.execute_buy('AAPL', shares=10000, market_price=150.0)
        
        # Cost should scale with order size
        assert cost > 400.0

    def test_execute_buy_small_order(self):
        """Test buy execution with small order."""
        config = ExecutionConfig()
        engine = ExecutionEngine(config)
        
        fill_price, cost = engine.execute_buy('AAPL', shares=1, market_price=150.0)
        
        # Cost should be small but non-zero
        assert cost > 0
        assert cost < 1.0


class TestExecutionEngineSell:
    """Test suite for ExecutionEngine.execute_sell method."""

    def test_execute_sell_basic(self):
        """Test basic sell execution with costs."""
        config = ExecutionConfig(transaction_cost_bps=3.0, slippage_bps=2.0)
        engine = ExecutionEngine(config)
        
        fill_price, cost = engine.execute_sell('AAPL', shares=100, market_price=150.0)
        
        # Slippage: 150 * 0.0002 = 0.03 per share
        # Fill price: 150 - 0.03 = 149.97 (we receive less when selling)
        expected_fill = 150.0 - (150.0 * 0.0002)
        assert pytest.approx(fill_price, rel=1e-6) == expected_fill
        
        # Transaction cost: 100 * 149.97 * 0.0003
        trade_value = 100 * fill_price
        expected_cost = trade_value * 0.0003
        assert pytest.approx(cost, rel=1e-6) == expected_cost

    def test_execute_sell_zero_costs(self):
        """Test sell execution with zero transaction costs."""
        config = ExecutionConfig(transaction_cost_bps=0.0, slippage_bps=0.0)
        engine = ExecutionEngine(config)
        
        fill_price, cost = engine.execute_sell('AAPL', shares=100, market_price=150.0)
        
        assert fill_price == 150.0  # No slippage
        assert cost == 0.0  # No transaction cost

    def test_execute_sell_asymmetry(self):
        """Test that sell slippage is opposite of buy slippage."""
        config = ExecutionConfig(slippage_bps=10.0)
        engine = ExecutionEngine(config)
        
        buy_fill, _ = engine.execute_buy('AAPL', 100, 100.0)
        sell_fill, _ = engine.execute_sell('AAPL', 100, 100.0)
        
        # Buy should have positive slippage, sell negative
        assert buy_fill > 100.0
        assert sell_fill < 100.0
        assert pytest.approx(buy_fill - 100.0, rel=1e-6) == 100.0 - sell_fill

    def test_execute_sell_large_order(self):
        """Test sell execution with large order."""
        config = ExecutionConfig()
        engine = ExecutionEngine(config)
        
        fill_price, cost = engine.execute_sell('AAPL', shares=10000, market_price=150.0)
        
        # Fill price should have slippage applied
        assert fill_price < 150.0
        # Cost should scale with order size
        assert cost > 400.0


class TestExecutionEngineShouldExecute:
    """Test suite for ExecutionEngine.should_execute method."""

    def test_should_execute_above_minimum(self):
        """Test trade value above minimum threshold."""
        config = ExecutionConfig(min_trade_value=100.0)
        engine = ExecutionEngine(config)
        
        # 10 shares * $15 = $150 > $100
        assert engine.should_execute(shares=10, price=15.0) is True

    def test_should_execute_below_minimum(self):
        """Test trade value below minimum threshold."""
        config = ExecutionConfig(min_trade_value=100.0)
        engine = ExecutionEngine(config)
        
        # 5 shares * $10 = $50 < $100
        assert engine.should_execute(shares=5, price=10.0) is False

    def test_should_execute_exact_minimum(self):
        """Test trade value exactly at minimum threshold."""
        config = ExecutionConfig(min_trade_value=100.0)
        engine = ExecutionEngine(config)
        
        # 10 shares * $10 = $100 == $100
        assert engine.should_execute(shares=10, price=10.0) is True

    def test_should_execute_zero_minimum(self):
        """Test with zero minimum trade value."""
        config = ExecutionConfig(min_trade_value=0.0)
        engine = ExecutionEngine(config)
        
        # Any positive value should execute
        assert engine.should_execute(shares=1, price=0.01) is True

    def test_should_execute_high_price_low_shares(self):
        """Test expensive stock with few shares."""
        config = ExecutionConfig(min_trade_value=1000.0)
        engine = ExecutionEngine(config)
        
        # 1 share * $2000 = $2000 > $1000
        assert engine.should_execute(shares=1, price=2000.0) is True

    def test_should_execute_low_price_high_shares(self):
        """Test cheap stock with many shares."""
        config = ExecutionConfig(min_trade_value=1000.0)
        engine = ExecutionEngine(config)
        
        # 10000 shares * $0.05 = $500 < $1000
        assert engine.should_execute(shares=10000, price=0.05) is False


class TestExecutionEngineMarketImpact:
    """Test suite for ExecutionEngine.calculate_market_impact method."""

    def test_market_impact_zero_factor(self):
        """Test market impact with zero factor (disabled)."""
        config = ExecutionConfig(market_impact_factor=0.0)
        engine = ExecutionEngine(config)
        
        impact = engine.calculate_market_impact(
            shares=1000,
            price=100.0,
            avg_daily_volume=1000000
        )
        
        assert impact == 0.0

    def test_market_impact_small_trade(self):
        """Test market impact for small trade relative to volume."""
        config = ExecutionConfig(market_impact_factor=0.01)
        engine = ExecutionEngine(config)
        
        # 100 shares of $100 stock, 1M daily volume
        impact = engine.calculate_market_impact(
            shares=100,
            price=100.0,
            avg_daily_volume=1000000
        )
        
        # Should be very small
        assert impact >= 0
        assert impact < 1.0

    def test_market_impact_large_trade(self):
        """Test market impact for large trade relative to volume."""
        config = ExecutionConfig(market_impact_factor=0.01)
        engine = ExecutionEngine(config)
        
        # 10000 shares of $100 stock, 1M daily volume
        impact = engine.calculate_market_impact(
            shares=10000,
            price=100.0,
            avg_daily_volume=1000000
        )
        
        # Should be larger for bigger trade
        assert impact > 0

    def test_market_impact_zero_volume(self):
        """Test market impact with zero daily volume (edge case)."""
        config = ExecutionConfig(market_impact_factor=0.01)
        engine = ExecutionEngine(config)
        
        impact = engine.calculate_market_impact(
            shares=1000,
            price=100.0,
            avg_daily_volume=0
        )
        
        # Should return 0 to avoid division by zero
        assert impact == 0.0

    def test_market_impact_scaling(self):
        """Test that market impact scales with trade size."""
        config = ExecutionConfig(market_impact_factor=0.01)
        engine = ExecutionEngine(config)
        
        small_impact = engine.calculate_market_impact(100, 100.0, 1000000)
        large_impact = engine.calculate_market_impact(10000, 100.0, 1000000)
        
        # Larger trade should have larger impact
        assert large_impact > small_impact


class TestExecutionEngineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_buy_and_sell_roundtrip(self):
        """Test that buy and sell have symmetric but opposite costs."""
        config = ExecutionConfig(transaction_cost_bps=5.0, slippage_bps=3.0)
        engine = ExecutionEngine(config)
        
        market_price = 100.0
        shares = 100
        
        buy_fill, buy_cost = engine.execute_buy('AAPL', shares, market_price)
        sell_fill, sell_cost = engine.execute_sell('AAPL', shares, market_price)
        
        # Buy fill should be higher, sell fill should be lower
        assert buy_fill > market_price
        assert sell_fill < market_price
        
        # Costs should be roughly equal (both based on trade value)
        assert pytest.approx(buy_cost, rel=0.01) == sell_cost

    def test_fractional_shares(self):
        """Test execution with fractional shares."""
        config = ExecutionConfig()
        engine = ExecutionEngine(config)
        
        fill_price, cost = engine.execute_buy('AAPL', shares=10.5, market_price=150.0)
        
        # Should handle fractional shares without error
        assert fill_price > 0
        assert cost > 0

    def test_very_small_price(self):
        """Test execution with very small price (penny stock)."""
        config = ExecutionConfig()
        engine = ExecutionEngine(config)
        
        fill_price, cost = engine.execute_buy('PENNY', shares=10000, market_price=0.01)
        
        # Should handle small prices
        assert fill_price > 0
        assert cost >= 0

    def test_very_large_price(self):
        """Test execution with very large price (BRK.A style)."""
        config = ExecutionConfig()
        engine = ExecutionEngine(config)
        
        fill_price, cost = engine.execute_buy('BRK.A', shares=1, market_price=500000.0)
        
        # Should handle large prices
        assert fill_price > 500000.0
        assert cost > 0

    def test_different_tickers_same_execution(self):
        """Test that ticker name doesn't affect execution."""
        config = ExecutionConfig()
        engine = ExecutionEngine(config)
        
        fill1, cost1 = engine.execute_buy('AAPL', 100, 150.0)
        fill2, cost2 = engine.execute_buy('MSFT', 100, 150.0)
        
        # Same shares and price should give same results regardless of ticker
        assert fill1 == fill2
        assert cost1 == cost2
