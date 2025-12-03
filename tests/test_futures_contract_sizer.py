"""
Unit tests for FuturesContractSizer.

Tests:
- Integer contract calculation
- Contract multiplier handling
- Position size constraints
- Edge cases

Run with: pytest tests/test_futures_contract_sizer.py -v
"""

import pytest
import numpy as np
from core.portfolio.position_sizers import FuturesContractSizer


class TestFuturesContractSizer:
    """Test FuturesContractSizer for integer contract position sizing."""
    
    @pytest.fixture
    def contract_multipliers(self):
        """Standard futures contract multipliers."""
        return {
            'ES': 50,      # S&P 500 E-mini
            'NQ': 20,      # Nasdaq 100 E-mini
            'CL': 1000,    # Crude Oil
            'NG': 10000,   # Natural Gas
            'GC': 100      # Gold
        }
    
    @pytest.fixture
    def sizer(self, contract_multipliers):
        """Create FuturesContractSizer instance."""
        return FuturesContractSizer(
            contract_multipliers=contract_multipliers,
            max_position_pct=0.25,
            risk_per_trade=0.02,
            min_contracts=1
        )
    
    def test_es_contract_calculation(self, sizer):
        """Test ES contract calculation with typical values."""
        # ES at 4500, portfolio $1M, 25% max = $250K max position
        # Notional per contract: 4500 × 50 = $225,000
        # Contracts: $250K / $225K = 1.11 → floor to 1
        
        contracts = sizer.calculate_size(
            ticker='ES',
            signal=1.0,
            current_price=4500.0,
            portfolio_value=1_000_000
        )
        
        assert contracts == 1.0
        assert contracts == int(contracts)  # Must be integer
    
    def test_cl_contract_calculation(self, sizer):
        """Test CL (crude oil) contract calculation."""
        # CL at $70, portfolio $1M, 25% max = $250K
        # Notional per contract: 70 × 1000 = $70,000
        # Contracts: $250K / $70K = 3.57 → floor to 3
        
        contracts = sizer.calculate_size(
            ticker='CL',
            signal=1.0,
            current_price=70.0,
            portfolio_value=1_000_000
        )
        
        assert contracts == 3.0
        assert contracts == int(contracts)
    
    def test_ng_contract_calculation(self, sizer):
        """Test NG (natural gas) contract calculation."""
        # NG at $3, portfolio $500K, 25% max = $125K
        # Notional per contract: 3 × 10000 = $30,000
        # Contracts: $125K / $30K = 4.16 → floor to 4
        
        contracts = sizer.calculate_size(
            ticker='NG',
            signal=1.0,
            current_price=3.0,
            portfolio_value=500_000
        )
        
        assert contracts == 4.0
        assert contracts == int(contracts)
    
    def test_signal_strength_scaling(self, sizer):
        """Test that signal strength scales position size."""
        # Full signal
        contracts_full = sizer.calculate_size(
            ticker='CL',
            signal=1.0,
            current_price=70.0,
            portfolio_value=1_000_000
        )
        
        # Half signal
        contracts_half = sizer.calculate_size(
            ticker='CL',
            signal=0.5,
            current_price=70.0,
            portfolio_value=1_000_000
        )
        
        # Half signal should give approximately half contracts (rounded down)
        assert contracts_half <= contracts_full / 2
        assert contracts_half == int(contracts_half)
    
    def test_minimum_contracts(self, sizer):
        """Test minimum contracts requirement."""
        # Small portfolio, expensive asset → fractional contract
        # Should return 0 if below minimum
        
        contracts = sizer.calculate_size(
            ticker='ES',
            signal=1.0,
            current_price=4500.0,
            portfolio_value=100_000  # Only $100K
        )
        
        # $100K × 25% = $25K, ES contract = $225K
        # Fractional = 0.11 → rounds to 0 (below min_contracts=1)
        assert contracts == 0.0
    
    def test_zero_signal_returns_zero(self, sizer):
        """Test that zero signal returns zero contracts."""
        contracts = sizer.calculate_size(
            ticker='CL',
            signal=0.0,
            current_price=70.0,
            portfolio_value=1_000_000
        )
        
        assert contracts == 0.0
    
    def test_weak_signal_below_minimum(self, sizer):
        """Test that weak signals below threshold return zero."""
        # Signal strength < 0.01 should return 0
        contracts = sizer.calculate_size(
            ticker='CL',
            signal=0.005,  # Very weak signal
            current_price=70.0,
            portfolio_value=1_000_000
        )
        
        assert contracts == 0.0
    
    def test_negative_signal_same_as_positive(self, sizer):
        """Test that negative signal (short) gives same size as positive."""
        contracts_long = sizer.calculate_size(
            ticker='CL',
            signal=1.0,
            current_price=70.0,
            portfolio_value=1_000_000
        )
        
        contracts_short = sizer.calculate_size(
            ticker='CL',
            signal=-1.0,
            current_price=70.0,
            portfolio_value=1_000_000
        )
        
        assert contracts_short == contracts_long
    
    def test_zero_price_returns_zero(self, sizer):
        """Test edge case: zero price."""
        contracts = sizer.calculate_size(
            ticker='CL',
            signal=1.0,
            current_price=0.0,
            portfolio_value=1_000_000
        )
        
        assert contracts == 0.0
    
    def test_zero_portfolio_returns_zero(self, sizer):
        """Test edge case: zero portfolio value."""
        contracts = sizer.calculate_size(
            ticker='CL',
            signal=1.0,
            current_price=70.0,
            portfolio_value=0.0
        )
        
        assert contracts == 0.0
    
    def test_unknown_ticker_uses_multiplier_1(self):
        """Test that unknown ticker defaults to multiplier 1 (like stocks)."""
        sizer = FuturesContractSizer(
            contract_multipliers={'CL': 1000},
            max_position_pct=0.25
        )
        
        # Unknown ticker should use multiplier=1
        contracts = sizer.calculate_size(
            ticker='UNKNOWN',
            signal=1.0,
            current_price=100.0,
            portfolio_value=1_000_000
        )
        
        # $1M × 25% = $250K, price=$100, multiplier=1
        # Contracts: $250K / ($100 × 1) = 2500
        assert contracts == 2500.0
    
    def test_large_portfolio_many_contracts(self, sizer):
        """Test calculation with large portfolio."""
        # $100M portfolio with NG at $3
        # Max position: $100M × 25% = $25M
        # Notional per contract: $3 × 10,000 = $30K
        # Contracts: $25M / $30K = 833
        
        contracts = sizer.calculate_size(
            ticker='NG',
            signal=1.0,
            current_price=3.0,
            portfolio_value=100_000_000
        )
        
        assert contracts == 833.0
        assert contracts == int(contracts)
    
    def test_contracts_always_integer(self, sizer):
        """Test that all outputs are integers (no fractions)."""
        test_cases = [
            ('ES', 4500.0, 1_000_000),
            ('NQ', 15000.0, 2_000_000),
            ('CL', 70.0, 500_000),
            ('NG', 3.5, 750_000),
            ('GC', 2000.0, 1_500_000)
        ]
        
        for ticker, price, portfolio in test_cases:
            contracts = sizer.calculate_size(
                ticker=ticker,
                signal=1.0,
                current_price=price,
                portfolio_value=portfolio
            )
            
            assert contracts == int(contracts), f"{ticker}: {contracts} is not integer"
            assert contracts >= 0, f"{ticker}: negative contracts {contracts}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
