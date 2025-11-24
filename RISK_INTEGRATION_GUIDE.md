# Risk Manager Integration Guide

## How Risk Manager Works with Backtest and Live Trading

This document explains how the RiskManager integrates with both backtesting and live trading systems.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Portfolio System                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │   Backtest   │      │ Live Trading │                    │
│  │   Engine     │      │   Engine     │                    │
│  └──────┬───────┘      └──────┬───────┘                    │
│         │                     │                             │
│         │    ┌────────────────┴──────────────┐             │
│         │    │                                │             │
│         └───►│        RiskManager             │◄────────┐   │
│              │                                │         │   │
│              │  • Position Sizing             │         │   │
│              │  • Pre-Trade Validation        │         │   │
│              │  • Risk Metrics Tracking       │         │   │
│              │  • Violation Logging           │         │   │
│              └────────────┬───────────────────┘         │   │
│                           │                             │   │
│                           ▼                             │   │
│                  ┌─────────────────┐                   │   │
│                  │  RiskDashboard  │                   │   │
│                  │  (Visualization)│                   │   │
│                  └─────────────────┘                   │   │
│                                                         │   │
│              ┌──────────────────────────────────┐      │   │
│              │     Data Sources                  │      │   │
│              │  • Historical Prices              │──────┘   │
│              │  • Live Market Data               │          │
│              │  • Position Tracking              │          │
│              └──────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### 1. Backtest Integration

**Location:** `core/portfolio_manager.py`

**Current State:** RiskManager is standalone (not yet integrated)

**Integration Plan:**

#### A. Add RiskManager to PortfolioConfig

```python
@dataclass
class PortfolioConfig:
    initial_capital: float = 100000.0
    rebalance_threshold: float = 0.05
    transaction_cost_bps: float = 3.0
    
    # NEW: Optional risk manager
    risk_manager: Optional[RiskManager] = None
```

#### B. Modify Backtest Loop

**Before each trade:**

```python
def _run_backtest(signals_dict, prices_dict, config):
    # ... existing setup ...
    
    risk_mgr = config.risk_manager  # Get risk manager if provided
    
    for date in trading_dates:
        # ... existing code ...
        
        # NEW: Update risk manager with daily returns
        if risk_mgr:
            for ticker in active_tickers:
                daily_return = calculate_daily_return(ticker, date)
                risk_mgr.update_returns(ticker, date, daily_return)
            
            # Update correlations periodically (e.g., weekly)
            if should_update_correlations(date):
                returns_df = get_recent_returns(prices_dict)
                risk_mgr.update_correlations(returns_df)
        
        # For each signal change
        for ticker, signal in new_signals.items():
            if signal != 0:  # Want to take position
                
                # NEW: Calculate position size with risk manager
                if risk_mgr:
                    # Get current volatility
                    recent_returns = get_recent_returns_for_ticker(ticker)
                    volatility = risk_mgr.calculate_volatility(ticker, recent_returns)
                    
                    # Get historical stats for Kelly (if available)
                    win_rate, avg_win, avg_loss = calculate_trade_stats(ticker)
                    
                    # Calculate optimal position size
                    pos_size = risk_mgr.calculate_position_size(
                        ticker=ticker,
                        signal=signal,
                        capital=current_cash,
                        positions=current_positions,
                        volatility=volatility,
                        win_rate=win_rate,
                        avg_win=avg_win,
                        avg_loss=avg_loss
                    )
                else:
                    # Default: equal weight
                    pos_size = default_position_size
                
                # Calculate shares to trade
                target_value = current_portfolio_value * pos_size
                shares = calculate_shares(target_value, current_price)
                
                # NEW: Validate trade with risk manager
                if risk_mgr:
                    is_valid, reason = risk_mgr.validate_trade(
                        ticker=ticker,
                        size=pos_size,
                        positions=current_positions,
                        portfolio_value=current_portfolio_value,
                        prices=current_prices
                    )
                    
                    if not is_valid:
                        print(f"Trade rejected for {ticker}: {reason}")
                        continue  # Skip this trade
                
                # Execute trade (existing code)
                execute_trade(ticker, shares, price)
        
        # NEW: Check stop conditions
        if risk_mgr:
            current_dd = calculate_current_drawdown(equity_curve)
            should_stop, reason = risk_mgr.check_stop_conditions(
                current_drawdown=current_dd,
                equity_curve=equity_curve
            )
            
            if should_stop:
                print(f"Trading stopped: {reason}")
                break  # Exit backtest
        
        # NEW: Log risk metrics
        if risk_mgr:
            current_dd = calculate_current_drawdown(equity_curve)
            risk_mgr.log_metrics(
                date=date,
                positions=current_positions,
                prices=current_prices,
                portfolio_value=current_portfolio_value,
                drawdown=current_dd
            )
    
    # NEW: Return risk metrics with results
    if risk_mgr:
        return (result, equity_df, trades_df, 
                risk_mgr.get_metrics_dataframe(), 
                risk_mgr.get_violations_dataframe())
    else:
        return result, equity_df, trades_df
```

#### C. Usage Example

```python
# Create risk manager
risk_config = RiskConfig(
    position_sizing_method='vol_adjusted',
    max_position_size=0.25,
    max_leverage=1.0,
    volatility_target=0.15,
    max_drawdown_stop=-0.20
)
risk_mgr = RiskManager(risk_config)

# Configure portfolio with risk management
config = PortfolioConfig(
    initial_capital=100000,
    rebalance_threshold=0.05,
    transaction_cost_bps=3.0,
    risk_manager=risk_mgr  # Add risk manager
)

# Run backtest (risk manager works automatically)
result, equity_df, trades_df, risk_metrics_df, violations_df = run_multi_asset_backtest(
    signals_dict=signals,
    prices_dict=prices,
    config=config,
    return_pm=False
)

# Generate risk dashboard
dashboard = RiskDashboard()
dashboard.generate_dashboard(
    risk_metrics_df=risk_metrics_df,
    violations_df=violations_df,
    correlation_matrix=risk_mgr.correlation_matrix,
    save_path='reports/risk_dashboard.html'
)
```

---

### 2. Live Trading Integration

**Location:** `live/run_strategy.py`

**Integration Points:**

#### A. Initialize Risk Manager at Startup

```python
class LiveTrader:
    def __init__(self, config):
        self.config = config
        
        # Initialize risk manager
        risk_config = RiskConfig(
            position_sizing_method='vol_adjusted',
            max_position_size=0.20,
            max_leverage=1.0,
            volatility_target=0.15,
            max_drawdown_stop=-0.15,
            correlation_window=60
        )
        self.risk_mgr = RiskManager(risk_config)
        
        # Load historical data for initial calculations
        self._initialize_risk_manager()
    
    def _initialize_risk_manager(self):
        """Pre-populate risk manager with recent history."""
        # Load last 60 days of data
        recent_prices = load_recent_prices(days=60)
        
        # Calculate returns and update correlations
        returns_df = calculate_returns(recent_prices)
        self.risk_mgr.update_correlations(returns_df)
        
        # Pre-calculate volatilities
        for ticker in self.tickers:
            vol = returns_df[ticker].std() * np.sqrt(252)
            self.risk_mgr.volatility_cache[ticker] = vol
```

#### B. Pre-Trade Validation

```python
def execute_signal(self, ticker, signal, current_price):
    """Execute a trading signal with risk management."""
    
    # Get current portfolio state
    portfolio_value = self.get_portfolio_value()
    positions = self.get_current_positions()
    
    # Calculate position size with risk manager
    pos_size = self.risk_mgr.calculate_position_size(
        ticker=ticker,
        signal=signal,
        capital=self.get_available_cash(),
        positions=positions,
        volatility=self.risk_mgr.calculate_volatility(ticker)
    )
    
    # Validate trade
    current_prices = self.get_current_prices()
    is_valid, reason = self.risk_mgr.validate_trade(
        ticker=ticker,
        size=pos_size,
        positions=positions,
        portfolio_value=portfolio_value,
        prices=current_prices
    )
    
    if not is_valid:
        logger.warning(f"Trade rejected for {ticker}: {reason}")
        self.log_violation(ticker, reason)
        return None
    
    # Calculate shares to trade
    target_value = portfolio_value * pos_size
    shares = int(target_value / current_price)
    
    # Execute the trade
    order = self.broker.submit_order(ticker, shares, 'market')
    logger.info(f"Order submitted: {ticker} {shares} shares @ ${current_price}")
    
    return order
```

#### C. Daily Risk Monitoring

```python
def daily_risk_check(self):
    """Run daily risk checks and update metrics."""
    
    # Update portfolio state
    portfolio_value = self.get_portfolio_value()
    positions = self.get_current_positions()
    prices = self.get_current_prices()
    
    # Calculate current drawdown
    equity_history = self.get_equity_history()
    peak = equity_history.max()
    current_dd = (portfolio_value - peak) / peak
    
    # Check stop conditions
    should_stop, reason = self.risk_mgr.check_stop_conditions(
        current_drawdown=current_dd,
        equity_curve=equity_history
    )
    
    if should_stop:
        logger.critical(f"RISK STOP TRIGGERED: {reason}")
        self.emergency_liquidate()
        self.send_alert(f"Trading stopped: {reason}")
        return False
    
    # Log risk metrics
    self.risk_mgr.log_metrics(
        date=pd.Timestamp.now(),
        positions=positions,
        prices=prices,
        portfolio_value=portfolio_value,
        drawdown=current_dd
    )
    
    # Update correlations (weekly)
    if datetime.now().weekday() == 0:  # Monday
        recent_returns = self.get_recent_returns(days=60)
        self.risk_mgr.update_correlations(recent_returns)
    
    # Check for warnings
    risk_metrics = self.risk_mgr.calculate_portfolio_risk(
        positions=positions,
        prices=prices,
        portfolio_value=portfolio_value
    )
    
    if risk_metrics['leverage'] > 0.9:
        logger.warning(f"High leverage: {risk_metrics['leverage']:.2f}x")
    
    if risk_metrics['max_position_weight'] > 0.23:
        logger.warning(f"High concentration: {risk_metrics['max_position_weight']:.2%}")
    
    return True
```

#### D. Real-Time Dashboard

```python
def generate_live_dashboard(self):
    """Generate real-time risk dashboard."""
    
    # Get risk metrics
    risk_metrics_df = self.risk_mgr.get_metrics_dataframe()
    violations_df = self.risk_mgr.get_violations_dataframe()
    
    # Generate dashboard
    dashboard = RiskDashboard(output_dir='reports/live')
    
    dashboard.generate_dashboard(
        risk_metrics_df=risk_metrics_df,
        violations_df=violations_df,
        correlation_matrix=self.risk_mgr.correlation_matrix,
        title=f"Live Trading Risk Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        save_path='reports/live/current_risk.html'
    )
    
    # Update every hour during trading
    schedule.every().hour.do(self.generate_live_dashboard)
```

---

## Key Benefits of Integration

### Backtest Benefits:

1. **Realistic Position Sizing**
   - Accounts for volatility differences between assets
   - Prevents over-concentration
   - Optimizes for growth (Kelly) or risk (vol-adjusted)

2. **Risk Limit Enforcement**
   - Automatic rejection of oversized trades
   - Leverage control prevents over-exposure
   - Drawdown stops protect capital

3. **Historical Risk Analysis**
   - Track how risk evolved during backtest
   - Identify periods of high concentration
   - Correlation breakdown detection

4. **Better Strategy Comparison**
   - Compare strategies on risk-adjusted basis
   - See how sizing methods affect results
   - Understand leverage usage patterns

### Live Trading Benefits:

1. **Pre-Trade Safety Checks**
   - Every trade validated before execution
   - Prevents dangerous trades automatically
   - Violations logged for review

2. **Real-Time Monitoring**
   - Live leverage tracking
   - Current drawdown monitoring
   - Position concentration alerts

3. **Emergency Protection**
   - Automatic trading halt on max drawdown
   - Correlation spike detection
   - Forced liquidation capability

4. **Audit Trail**
   - Complete violation history
   - Risk metrics logged continuously
   - Dashboard for compliance/review

---

## Implementation Timeline

### Phase 1 ✅ (Completed)
- RiskManager core implementation
- Position sizing methods
- Risk validation
- Interactive dashboards

### Phase 2 (In Progress)
- **Backtest Integration** (1-2 days)
  - Add risk_manager to PortfolioConfig
  - Integrate sizing calculations
  - Add pre-trade validation
  - Return risk metrics with results

- **Demo Notebook** (1 day)
  - Show integrated backtest
  - Compare with/without risk management
  - Demonstrate violation handling

### Phase 3 (Future)
- **Live Trading Integration** (2-3 days)
  - Initialize risk manager with history
  - Pre-trade validation in live trader
  - Real-time monitoring
  - Emergency stop logic

- **Advanced Features** (Phase 2 features)
  - VaR/CVaR calculations
  - Regime detection
  - Liquidity risk
  - Behavioral alerts

---

## Testing Strategy

### Unit Tests
```python
def test_risk_manager_integration():
    # Test with risk manager
    config_with_risk = PortfolioConfig(
        initial_capital=100000,
        risk_manager=RiskManager(RiskConfig(max_position_size=0.20))
    )
    result_risk = run_backtest(signals, prices, config_with_risk)
    
    # Test without risk manager
    config_no_risk = PortfolioConfig(initial_capital=100000)
    result_no_risk = run_backtest(signals, prices, config_no_risk)
    
    # Risk-managed should have fewer/smaller trades
    assert len(result_risk.trades) <= len(result_no_risk.trades)
    assert result_risk.max_leverage <= 1.0
```

### Integration Tests
- Run full backtests with different risk configs
- Verify violations are logged correctly
- Test stop conditions trigger properly
- Ensure metrics collection doesn't slow backtest

### Live Trading Tests
- Paper trading with risk manager
- Test emergency stop procedure
- Verify real-time dashboard updates
- Test violation alert system

---

## Configuration Examples

### Conservative (Capital Preservation)
```python
risk_config = RiskConfig(
    position_sizing_method='fixed_fraction',
    fixed_fraction=0.01,  # Risk 1% per trade
    max_position_size=0.15,  # Max 15% per position
    max_leverage=0.8,  # Stay under-leveraged
    max_drawdown_stop=-0.10  # Stop at -10% DD
)
```

### Moderate (Balanced)
```python
risk_config = RiskConfig(
    position_sizing_method='vol_adjusted',
    volatility_target=0.12,  # 12% portfolio vol
    max_position_size=0.25,
    max_leverage=1.0,
    max_drawdown_stop=-0.20
)
```

### Aggressive (Growth)
```python
risk_config = RiskConfig(
    position_sizing_method='kelly',
    kelly_fraction=0.5,  # Half Kelly
    max_position_size=0.30,
    max_leverage=1.5,  # Moderate leverage
    max_drawdown_stop=-0.25
)
```

---

## Summary

The RiskManager integrates seamlessly with both backtesting and live trading:

**Backtesting:**
- Validates trades before execution
- Calculates optimal position sizes
- Tracks risk metrics throughout test
- Generates comprehensive risk reports

**Live Trading:**
- Real-time pre-trade validation
- Dynamic position sizing
- Continuous risk monitoring
- Emergency stop protection

**Next Step:** Implement backtest integration (Phase 2) to enable risk-managed backtests.
