"""
Test PortfolioManager logic with simple scenarios.
Verifies that allocation works correctly with signal-driven entries/exits.
"""
import pandas as pd
import numpy as np
from core.portfolio_manager import PortfolioManager, PortfolioConfig

def test_allocation_logic():
    """Test that allocation works as expected."""
    
    print("Testing Portfolio Manager Allocation Logic")
    print("="*60)
    
    config = PortfolioConfig(
        initial_capital=100000,
        rebalance_threshold=0.05,
        transaction_cost_bps=3.0
    )
    
    pm = PortfolioManager(config)
    
    # Test 1: Both signals active (should be 50/50)
    print("\n1. Both ES and GC signal LONG")
    print("-"*60)
    signals = {'ES': 1, 'GC': 1}
    weights = pm.calculate_target_allocation(signals)
    print(f"Target weights: {weights}")
    print(f"Expected: ES=0.5, GC=0.5")
    assert weights['ES'] == 0.5
    assert weights['GC'] == 0.5
    print("✓ PASS")
    
    # Test 2: Only ES active (should be 100% ES)
    print("\n2. ES signal LONG, GC signal FLAT")
    print("-"*60)
    signals = {'ES': 1, 'GC': 0}
    weights = pm.calculate_target_allocation(signals)
    print(f"Target weights: {weights}")
    print(f"Expected: ES=1.0, GC=0.0")
    assert weights['ES'] == 1.0
    assert weights['GC'] == 0.0
    print("✓ PASS")
    
    # Test 3: Only GC active (should be 100% GC)
    print("\n3. ES signal FLAT, GC signal LONG")
    print("-"*60)
    signals = {'ES': 0, 'GC': 1}
    weights = pm.calculate_target_allocation(signals)
    print(f"Target weights: {weights}")
    print(f"Expected: ES=0.0, GC=1.0")
    assert weights['ES'] == 0.0
    assert weights['GC'] == 1.0
    print("✓ PASS")
    
    # Test 4: No signals active (should be 0% each)
    print("\n4. Both ES and GC signal FLAT")
    print("-"*60)
    signals = {'ES': 0, 'GC': 0}
    weights = pm.calculate_target_allocation(signals)
    print(f"Target weights: {weights}")
    print(f"Expected: ES=0.0, GC=0.0")
    assert weights['ES'] == 0.0
    assert weights['GC'] == 0.0
    print("✓ PASS")
    
    # Test 5: Three assets, two active
    print("\n5. ES LONG, GC LONG, NQ FLAT")
    print("-"*60)
    signals = {'ES': 1, 'GC': 1, 'NQ': 0}
    weights = pm.calculate_target_allocation(signals)
    print(f"Target weights: {weights}")
    print(f"Expected: ES=0.5, GC=0.5, NQ=0.0")
    assert weights['ES'] == 0.5
    assert weights['GC'] == 0.5
    assert weights['NQ'] == 0.0
    print("✓ PASS")
    
    # Test 6: Three assets, all active
    print("\n6. ES LONG, GC LONG, NQ LONG")
    print("-"*60)
    signals = {'ES': 1, 'GC': 1, 'NQ': 1}
    weights = pm.calculate_target_allocation(signals)
    print(f"Target weights: {weights}")
    print(f"Expected: ES=0.33, GC=0.33, NQ=0.33")
    assert abs(weights['ES'] - 1/3) < 0.001
    assert abs(weights['GC'] - 1/3) < 0.001
    assert abs(weights['NQ'] - 1/3) < 0.001
    print("✓ PASS")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
    
    # Demonstrate the workflow
    print("\n\nWorkflow Example:")
    print("="*60)
    print("\nDay 1: ES Long, GC Flat")
    print("  → Allocate 100% to ES ($100k)")
    print("  → GC gets $0")
    
    print("\nDay 50: ES Long (still), GC Long (NEW signal)")
    print("  → Current: ES=$120k (from 20% gain)")
    print("  → Action: Rebalance among active positions")
    print("  → Sell $30k ES, Buy $60k GC")
    print("  → Result: ES=$90k, GC=$60k")
    print("  → Wait, that's $150k invested! Wrong!")
    print("\n  CORRECT Action:")
    print("  → Total active capital: $120k")
    print("  → Split 50/50: ES=$60k, GC=$60k")
    print("  → Sell $60k ES, Buy $60k GC")
    print("  → Result: ES=$60k, GC=$60k ✓")
    
    print("\nDay 100: ES Long, GC Long (both still active)")
    print("  → ES grows to $80k, GC stays at $60k")
    print("  → Total active: $140k")
    print("  → ES weight: 80/140 = 57%, GC weight: 60/140 = 43%")
    print("  → Drift: |57%-50%| = 7% > 5% threshold")
    print("  → Action: Rebalance")
    print("  → Sell $10k ES, Buy $10k GC")
    print("  → Result: ES=$70k (50%), GC=$70k (50%) ✓")
    
    print("\nDay 150: ES Long, GC Exit (signal says close)")
    print("  → Close GC position entirely")
    print("  → All capital goes to ES (only active signal)")
    print("  → No rebalancing needed (only 1 active position)")

if __name__ == "__main__":
    test_allocation_logic()
