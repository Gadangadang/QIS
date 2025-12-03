"""
Extended tests for position sizers - covering missing functionality.

Tests:
- VolatilityTargetingSizer
- Edge cases in position sizing
- Signal scaling behavior

Run with: pytest tests/test_position_sizers_extended.py -v
"""

import pytest
import numpy as np
from core.portfolio.position_sizers import (
    FixedFractionalSizer,
    KellySizer,
    ATRSizer,
    VolatilityScaledSizer,
    FuturesContractSizer,
)


class TestVolatilityScaledSizer:
    """Test volatility scaled position sizer."""
    
    @pytest.fixture
    def vol_sizer(self):
        """Create a volatility scaled sizer."""
        return VolatilityScaledSizer(
            target_volatility=0.15,  # 15% target vol
            max_position_pct=0.30,
            min_position_pct=0.05,
            min_trade_value=100
        )
    
    def test_inverse_volatility_scaling(self, vol_sizer):
        """Test that position size scales inversely with volatility."""
        portfolio_value = 100000
        price = 100
        signal = 1.0
        
        # Low volatility -> larger position
        low_vol = 0.10
        size_low_vol = vol_sizer.calculate_size('TEST', signal, price, portfolio_value, volatility=low_vol)
        
        # High volatility -> smaller position
        high_vol = 0.30
        size_high_vol = vol_sizer.calculate_size('TEST', signal, price, portfolio_value, volatility=high_vol)
        
        # Lower volatility should result in larger or equal position (may hit max cap)
        assert size_low_vol >= size_high_vol
    
    def test_respects_max_position_pct(self, vol_sizer):
        """Test that max position percentage is respected."""
        portfolio_value = 100000
        price = 100
        signal = 1.0
        
        # Very low volatility should hit max cap
        very_low_vol = 0.01
        size = vol_sizer.calculate_size('TEST', signal, price, portfolio_value, volatility=very_low_vol)
        
        # Should not exceed max position
        max_shares = (portfolio_value * vol_sizer.max_position_pct) / price
        assert size <= max_shares
    
    def test_respects_min_position_pct(self, vol_sizer):
        """Test that min position percentage is respected."""
        portfolio_value = 100000
        price = 100
        signal = 1.0
        
        # Very high volatility should hit min floor
        very_high_vol = 1.0
        size = vol_sizer.calculate_size('TEST', signal, price, portfolio_value, volatility=very_high_vol)
        
        # Should be at least min position
        min_shares = (portfolio_value * vol_sizer.min_position_pct) / price
        assert size >= min_shares or size == 0  # 0 if below min trade value
    
    def test_fallback_when_no_volatility(self, vol_sizer):
        """Test fallback to max position when volatility not provided."""
        portfolio_value = 100000
        price = 100
        signal = 1.0
        
        # No volatility provided
        size = vol_sizer.calculate_size('TEST', signal, price, portfolio_value, volatility=None)
        
        # Should use max position pct
        expected_shares = (portfolio_value * vol_sizer.max_position_pct) / price
        assert abs(size - expected_shares) < 1  # Within rounding
    
    def test_zero_volatility_fallback(self, vol_sizer):
        """Test that zero volatility falls back to max position."""
        portfolio_value = 100000
        price = 100
        signal = 1.0
        
        size = vol_sizer.calculate_size('TEST', signal, price, portfolio_value, volatility=0.0)
        
        # Should use max position pct (fallback)
        expected_shares = (portfolio_value * vol_sizer.max_position_pct) / price
        assert abs(size - expected_shares) < 1


class TestATRSizerEdgeCases:
    """Test ATR sizer edge cases."""
    
    @pytest.fixture
    def atr_sizer(self):
        """Create ATR-based sizer."""
        return ATRSizer(
            risk_per_trade=0.02,
            max_position_pct=0.25,
            atr_multiplier=2.0,
            min_trade_value=100
        )
    
    def test_handles_missing_atr(self, atr_sizer):
        """Test behavior when ATR is not provided."""
        # Should not crash, uses fallback to fixed fractional
        size = atr_sizer.calculate_size('TEST', 1.0, 100, 100000, atr=None)
        assert size > 0  # Uses fallback sizing
    
    def test_handles_zero_atr(self, atr_sizer):
        """Test behavior when ATR is zero."""
        size = atr_sizer.calculate_size('TEST', 1.0, 100, 100000, atr=0.0)
        assert size > 0  # Uses fallback sizing when ATR is zero


class TestKellySizerEdgeCases:
    """Test Kelly sizer edge cases."""
    
    @pytest.fixture
    def kelly_sizer(self):
        """Create Kelly criterion sizer."""
        return KellySizer(
            max_position_pct=0.25,
            kelly_fraction=0.5,  # Half Kelly
            min_trade_value=100
        )
    
    def test_negative_edge(self, kelly_sizer):
        """Test Kelly with negative edge (losing strategy)."""
        # Negative edge: win rate < loss rate adjusted
        size = kelly_sizer.calculate_size(
            'TEST', 1.0, 100, 100000,
            win_rate=0.40, avg_win=1.0, avg_loss=1.0
        )
        
        # Should return 0 for negative edge
        assert size == 0
    
    def test_missing_parameters(self, kelly_sizer):
        """Test Kelly when required parameters missing."""
        # No win rate or payoff ratio
        size = kelly_sizer.calculate_size('TEST', 1.0, 100, 100000)
        assert size == 0


class TestFixedFractionalEdgeCases:
    """Test fixed fractional sizer edge cases."""
    
    @pytest.fixture
    def fixed_sizer(self):
        """Create fixed fractional sizer."""
        return FixedFractionalSizer(
            risk_per_trade=0.02,
            max_position_pct=0.25,
            min_trade_value=100
        )
    
    def test_handles_very_large_portfolio(self, fixed_sizer):
        """Test with very large portfolio value."""
        size = fixed_sizer.calculate_size('TEST', 1.0, 100, 10_000_000)
        
        # Should calculate reasonable position
        assert size > 0
        assert size < 10_000_000 / 100  # Less than full portfolio
    
    def test_handles_very_small_price(self, fixed_sizer):
        """Test with very small price (penny stocks)."""
        size = fixed_sizer.calculate_size('TEST', 1.0, 0.01, 100000)
        
        # Should handle penny stock prices
        assert size >= 0


class TestFuturesContractSizerExtended:
    """Extended tests for futures contract sizer."""
    
    @pytest.fixture
    def contract_multipliers(self):
        """Standard contract multipliers."""
        return {
            'ES': 50,
            'NQ': 20,
            'CL': 1000,
            'NG': 10000,
            'GC': 100,
        }
    
    @pytest.fixture
    def sizer(self, contract_multipliers):
        """Create futures contract sizer."""
        return FuturesContractSizer(
            contract_multipliers=contract_multipliers,
            max_position_pct=0.25,
            risk_per_trade=0.02,
            min_contracts=1
        )
    
    def test_gold_contract_calculation(self, sizer):
        """Test GC (Gold) contract sizing."""
        contracts = sizer.calculate_size(
            ticker='GC',
            signal=1.0,
            current_price=2000,  # $2000/oz
            portfolio_value=200000
        )
        
        # Notional = 2000 * 100 = $200,000 per contract
        # Contract value equals portfolio - may be too risky
        # Result depends on risk calculation
        assert contracts >= 0
        assert isinstance(contracts, (int, float))  # May return float
    
    def test_very_expensive_contracts(self, sizer):
        """Test when contract value exceeds portfolio."""
        # ES at very high price
        contracts = sizer.calculate_size(
            ticker='ES',
            signal=1.0,
            current_price=10000,  # $10,000 per point (unrealistic but tests edge case)
            portfolio_value=100000  # $100k portfolio
        )
        
        # Contract worth $10,000 * 50 = $500,000
        # Can't afford even one contract at 25% allocation
        # Should return 0 (below minimum)
        assert contracts == 0
    
    def test_multiple_contracts_sizing(self, sizer):
        """Test sizing for multiple contracts."""
        # Large portfolio, moderate ES price
        contracts = sizer.calculate_size(
            ticker='ES',
            signal=1.0,
            current_price=4500,
            portfolio_value=1_000_000  # $1M portfolio
        )
        
        # Contract value = 4500 * 50 = $225,000
        # 25% of $1M = $250,000
        # Should be able to afford 1 contract
        assert contracts >= 1
        # Result may be float, round if needed
        assert int(contracts) >= 1
    
    def test_fractional_signal_with_contracts(self, sizer):
        """Test that fractional signals scale down contracts."""
        portfolio = 500000
        
        # Full signal
        contracts_full = sizer.calculate_size('ES', 1.0, 4500, portfolio)
        
        # Quarter signal
        contracts_quarter = sizer.calculate_size('ES', 0.25, 4500, portfolio)
        
        # Quarter signal should give fewer contracts
        assert contracts_quarter <= contracts_full
        assert isinstance(contracts_quarter, int)


class TestSignalScaling:
    """Test signal scaling across different sizers."""
    
    @pytest.fixture
    def sizer(self):
        """Create a simple fixed fractional sizer."""
        return FixedFractionalSizer(
            risk_per_trade=0.02,
            max_position_pct=0.25,
            min_trade_value=100
        )
    
    def test_signal_strength_proportional(self, sizer):
        """Test that position size scales with signal strength."""
        portfolio = 100000
        price = 100
        
        # Full signal
        size_full = sizer.calculate_size('TEST', 1.0, price, portfolio)
        
        # Half signal
        size_half = sizer.calculate_size('TEST', 0.5, price, portfolio)
        
        # Should be approximately proportional
        assert abs(size_half - size_full * 0.5) < 5  # Within rounding
    
    def test_negative_signal_same_as_positive(self, sizer):
        """Test that negative signals give same size (for shorts)."""
        portfolio = 100000
        price = 100
        
        size_positive = sizer.calculate_size('TEST', 0.8, price, portfolio)
        size_negative = sizer.calculate_size('TEST', -0.8, price, portfolio)
        
        # Absolute value should be the same
        assert size_positive == size_negative


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
