# Strategic Plan: Professional-Grade Quantitative Trading System

**Date:** November 22, 2025  
**Objective:** Build a modular, institutional-quality quantitative trading system that demonstrates real-world quant expertise while remaining manageable for a solo developer.

---

## 📊 Current State Analysis

### ✅ What's Working Well

**Infrastructure (Week 0-1 Complete)**
- Clean, modular codebase with proper separation of concerns
- Vectorized execution engine (no loops in signal generation)
- Walk-forward validation with anchored windows (no lookahead bias)
- Comprehensive reporting framework with HTML generation
- Signal abstraction layer (`SignalModel` base class)
- Risk controls: stop-loss, take-profit, max-hold
- Parameter optimization framework (grid/random search)

**Solid Foundations**
- 35 years of S&P 500 data (1990-2025)
- Per-fold trade tracking with exit reasons
- Notebook-first research workflow
- Transaction cost modeling (3 bps)
- Regime-aware signals (bull market filters)

### ⚠️ Current Limitations

**Single-Asset, Single-Strategy**
- Only trades SPX (no multi-asset support)
- No portfolio-level risk management
- Always 100% invested (no position sizing)
- Can't run multiple strategies simultaneously
- No correlation analysis between strategies

**Performance Issues**
- Current strategy: Sharpe -0.33, CAGR -40%, MaxDD -99% ⚠️
- Low win rate (4-9%)
- Needs optimization (Week 2 partially done)

**Missing Components**
- No volatility targeting
- No Kelly criterion / fractional Kelly
- No futures support (critical for real-world quant)
- No intraday data capability
- No slippage modeling beyond flat transaction costs
- No borrow costs for shorts
- No contract rollover logic

---

## 🎯 End Goal Vision

### The Target System

A **multi-asset, multi-strategy portfolio manager** that:

1. **Runs 3-5 strategies simultaneously** across different asset classes
   - Equities strategy (SPY, QQQ, IWM)
   - Commodities strategy (Gold, Oil, Copper)
   - Fixed Income strategy (TLT, IEF)
   - Volatility strategy (VXX, VIXY)
   - FX strategy (EUR/USD, GBP/USD)

2. **Implements institutional-grade risk management**
   - Volatility targeting (e.g., 12% annualized)
   - Kelly-based position sizing
   - Strategy-level allocation (risk budgeting)
   - Correlation-aware diversification

3. **Handles realistic execution**
   - Futures contract rollovers
   - Bid-ask spreads
   - Market impact / slippage
   - Partial fills for large orders
   - Margin requirements

4. **Provides rigorous validation**
   - Walk-forward with per-fold optimization
   - Permutation tests (in-sample & walk-forward)
   - Monte Carlo regime shuffling
   - Statistical significance testing

---

## 🚀 Execution Roadmap

### Phase 1: Foundation Completion (Weeks 2-3) - 2 weeks
**Goal:** Complete optimization and add position sizing

#### Week 2: Optimization Excellence
- [x] Basic optimizer framework (DONE)
- [ ] **Per-fold optimization in walk-forward** (CRITICAL)
  - Optimize parameters on each training window
  - Use best params for that fold's test period
  - Track parameter evolution over time
- [ ] **Larger parameter grids**
  - Test 5x5 or larger grids (25-100 combinations)
  - Add more parameters: entry/exit thresholds, lookback windows
- [ ] **Regime-specific parameters**
  - Detect bull vs bear markets
  - Optimize separately for each regime
  - Switch parameters based on current regime

#### Week 3: Position Sizing & Risk Management
- [ ] **Volatility targeting**
  - Calculate realized volatility (e.g., 20-day rolling)
  - Scale position size inversely: `size = target_vol / realized_vol`
  - Target 10-12% annualized volatility
- [ ] **Kelly criterion**
  - Implement Kelly formula: `f* = (p*b - q) / b`
  - Add fractional Kelly (25-50% of full Kelly)
  - Backtest with Kelly vs fixed sizing
- [ ] **Dynamic bet sizing**
  - Size based on signal strength (e.g., momentum magnitude)
  - Implement min/max position caps
  - Add leverage constraints

**Expected Improvement:** Sharpe 0.5 → 1.0+, CAGR 0% → 8-12%

---

### Phase 2: Multi-Asset Framework (Weeks 4-5) - 2 weeks
**Goal:** Extend to futures and build portfolio manager

#### Week 4: Futures Infrastructure
- [ ] **Contract rollover logic**
  - Detect expiration dates
  - Roll from front to back month
  - Handle continuous price adjustments
- [ ] **Futures data loader**
  - ES (S&P 500), NQ (Nasdaq), RTY (Russell 2000)
  - GC (Gold), CL (Oil), HG (Copper)
  - ZN/ZB (Treasury notes/bonds)
- [ ] **Margin & leverage**
  - Calculate margin requirements per contract
  - Track available capital
  - Prevent over-leveraging

#### Week 5: Portfolio Manager
- [ ] **Multi-strategy orchestrator**
  - Run 3-5 strategies in parallel
  - Each strategy generates signals for its asset(s)
  - Aggregate into portfolio-level positions
- [ ] **Risk budgeting**
  - Allocate risk (not capital) across strategies
  - E.g., each strategy gets 3% vol target
  - Portfolio target: 12% vol = 4 strategies × 3% each
- [ ] **Correlation management**
  - Calculate rolling correlation matrix
  - Reduce allocations to highly correlated strategies
  - Increase diversification benefit

**Expected Improvement:** Portfolio Sharpe 1.0 → 1.5+, better drawdown profile

---

### Phase 3: Realistic Execution (Weeks 6-7) - 2 weeks
**Goal:** Model real-world costs and slippage

#### Week 6: Advanced Slippage Model
- [ ] **Bid-ask spread**
  - Model spread as % of price (wider for illiquid assets)
  - Wider spreads for large orders
  - Extra slippage for shorts (borrow costs)
- [ ] **Market impact**
  - Slippage proportional to order size
  - `slippage = k * sqrt(order_size / avg_volume)`
  - Penalize large orders heavily
- [ ] **Partial fills**
  - Orders filled over multiple days if too large
  - Track unfilled quantity
  - Pay spread on each partial fill

#### Week 7: Borrow Costs & Constraints
- [ ] **Short borrow costs**
  - Add daily fee for holding shorts (e.g., 1-5% annualized)
  - Higher costs for hard-to-borrow assets
  - Track cumulative borrow fees
- [ ] **Position limits**
  - Max % of portfolio per asset (e.g., 20%)
  - Max leverage (e.g., 2x for futures)
  - Respect margin calls
- [ ] **Realistic constraints**
  - Can't short more than available margin
  - Must close positions if margin < maintenance level

**Expected Impact:** Sharpe 1.5 → 1.2-1.3 (realistic costs eat 15-20%)

---

### Phase 4: Statistical Rigor (Weeks 8-9) - 2 weeks
**Goal:** Prove results are not overfit

#### Week 8: Permutation Testing
- [ ] **In-sample permutation test**
  - Shuffle returns 1000 times
  - Re-run strategy on shuffled data
  - If real Sharpe > 95th percentile → statistically significant
- [ ] **Walk-forward permutation**
  - Shuffle fold order (Monte Carlo)
  - Ensure results robust to time period selection
  - Check if best parameters stable across regimes
- [ ] **Regime shuffling**
  - Shuffle bull/bear market labels
  - Test if regime filter adds real value
  - Quantify regime-switching benefit

#### Week 9: Execution Analysis Framework
- [ ] **Signal quality decomposition**
  - Pure signal Sharpe (no costs)
  - vs. executed Sharpe (with costs)
  - Quantify "implementation shortfall"
- [ ] **Trade-level attribution**
  - Which trades added/subtracted value?
  - Are losing trades systematically different?
  - Identify regime/market conditions for wins/losses
- [ ] **Sensitivity analysis**
  - How robust to parameter changes?
  - What if transaction costs 2x higher?
  - Monte Carlo stress testing

**Deliverable:** Statistical report proving results not due to luck/overfitting

---

### Phase 5: Final Integration (Weeks 10-11) - 2 weeks
**Goal:** Assemble full system and generate final report

#### Week 10: System Integration
- [ ] **Master dashboard**
  - Portfolio-level equity curve
  - Strategy attribution (which strategy contributed most?)
  - Risk decomposition (which asset driving volatility?)
- [ ] **Auto-rebalancing**
  - Daily: check volatility targets, rebalance if needed
  - Weekly: re-optimize parameters on new data
  - Monthly: adjust strategy allocations based on recent performance
- [ ] **Data pipeline**
  - Automated data fetching (if allowed by compliance)
  - Handle missing data gracefully
  - Contract rollover automation

#### Week 11: Final Validation & Documentation
- [ ] **Full backtest report**
  - 35 years, all assets, all strategies
  - Walk-forward results (no in-sample cherry-picking)
  - Statistical significance tests
- [ ] **Go/No-Go decision framework**
  - Required: Sharpe > 1.0, CAGR > 8%, MaxDD < 30%
  - If pass → document as "production-ready"
  - If fail → diagnose root cause, iterate
- [ ] **Documentation**
  - System architecture diagram
  - API documentation for all modules
  - User guide for running daily paper trades

---

## 🔬 Novel Problems to Tackle

### Problem 1: **Regime Detection Without Lookahead**
**Challenge:** Most regime detection uses hindsight (e.g., "2008 was a bear market"). How do you detect regime changes in real-time without lookahead bias?

**Approach:**
- Use hidden Markov models (HMM) with rolling windows
- Train on past data, predict current regime
- Compare to simple rules (SMA200) to see if HMM adds value

**Why It Matters:** If you can detect regime switches early, you can switch strategies dynamically (mean reversion in choppy markets, momentum in trends).

---

### Problem 2: **Optimal Parameter Stability**
**Challenge:** Optimal parameters change over time. How do you adapt without overfitting to recent data?

**Approach:**
- Track parameter evolution across walk-forward folds
- If parameters stable → good signal
- If parameters drift wildly → signal fragile
- Use ensemble of parameters (e.g., average top 3 parameter sets)

**Why It Matters:** Proves your signal has a stable underlying edge, not just curve-fit to one regime.

---

### Problem 3: **Transaction Cost Optimization**
**Challenge:** High-frequency rebalancing eats returns. How do you balance signal quality vs. turnover?

**Approach:**
- Add turnover penalty to optimization objective
- `objective = sharpe - λ * turnover`
- Find optimal λ via cross-validation
- Compare vs. simple threshold rules (only trade if signal > X)

**Why It Matters:** Institutional quants face this constantly. Showing you can optimize this demonstrates real-world expertise.

---

### Problem 4: **Portfolio Construction with Non-Stationary Correlations**
**Challenge:** Asset correlations change over time (especially in crises). How do you build robust portfolios?

**Approach:**
- Use shrinkage estimators (Ledoit-Wolf)
- Robust covariance estimation (Minimum Covariance Determinant)
- Compare to naïve equal-weight allocation

**Why It Matters:** Shows you understand portfolio theory beyond textbooks.

---

## 🎯 Success Metrics

### Technical Milestones
- [ ] Multi-asset portfolio manager operational (5+ assets)
- [ ] Walk-forward Sharpe > 1.0 (realistic costs)
- [ ] Statistical significance: p < 0.05 on permutation tests
- [ ] Modular codebase: each component independently testable
- [ ] Full documentation: architecture + user guide

### Showcase Metrics (for portfolio/interviews)
- **Performance:** Sharpe 1.2+, CAGR 10-15%, MaxDD < 25%
- **Robustness:** Positive returns in 70%+ of walk-forward folds
- **Realism:** All costs modeled (transaction, slippage, borrow)
- **Rigor:** Permutation tests show statistical significance
- **Novelty:** Solved 2-3 of the "hard problems" above

---

## 🚦 Prioritization Framework

### Must-Have (Week 2-7)
1. Per-fold optimization in walk-forward
2. Volatility targeting position sizing
3. Multi-asset support (at least 3 assets)
4. Portfolio risk manager
5. Realistic execution costs

### Should-Have (Week 8-9)
6. Permutation testing
7. Regime detection (HMM or similar)
8. Statistical significance framework
9. Transaction cost optimization

### Nice-to-Have (Week 10-11)
10. Intraday data support
11. Machine learning signals (LSTM, XGBoost)
12. Automated data pipeline
13. Real-time dashboard

---

## 🎓 Why This Plan Works

### 1. Incremental Complexity
- Week 2-3: Improve single-asset system
- Week 4-5: Add multi-asset capability
- Week 6-7: Add realism
- Week 8-9: Prove statistical rigor
- Week 10-11: Polish and integrate

### 2. Demonstrates Real-World Expertise
- Walk-forward validation → no lookahead bias
- Permutation testing → not overfit
- Multi-asset portfolio → real quant problem
- Transaction cost optimization → institutional concern

### 3. Manageable Scope
- 11 weeks total (doable in 3 months part-time)
- Each phase builds on previous
- Can stop at any phase and still have working system

### 4. Novel Contributions
- Regime-aware optimization (not common in public repos)
- Parameter stability analysis (unique insight)
- Transaction cost/turnover tradeoff (real-world problem)

---

## 📝 Next Immediate Actions

### This Week (Week 2)
1. **Run optimizer with larger grid** (5x5 = 25 combinations)
   - Parameters: lookback (100, 150, 200, 250, 300), stop_loss (5%, 8%, 10%, 12%, 15%)
   - Expected runtime: ~30 minutes
   
2. **Implement per-fold optimization**
   - Modify `run_walk_forward()` to accept `optimize_per_fold=True`
   - For each fold: run grid search on training data, use best params on test
   - Compare vs. global optimization

3. **Test regime-specific parameters**
   - Split data into bull/bear periods
   - Optimize separately
   - See if different parameters work better in each regime

### Next Week (Week 3)
4. **Add volatility targeting**
   - Calculate 20-day realized vol
   - Scale positions: `size = 12% / realized_vol`
   - Backtest and compare vs. fixed sizing

5. **Implement Kelly criterion**
   - Calculate from win rate and win/loss ratio
   - Test fractional Kelly (25%, 50%, 75%)
   - Compare Sharpe and drawdown profiles

---

## 🤔 Open Questions for Discussion

1. **Asset Selection:** Should we focus on liquid ETFs (SPY, QQQ) or move to futures (ES, NQ)?
   - ETFs: easier data, simpler, good for showcase
   - Futures: more institutional, better leverage, realistic for real quant fund

2. **Strategy Diversity:** Which asset classes prioritize?
   - Equities + Commodities + Fixed Income?
   - Or Equities + Volatility + FX?

3. **Signal Complexity:** Stick with simple signals (momentum, mean reversion) or add ML?
   - Simple: easier to explain, more interpretable
   - ML: more impressive, but harder to validate

4. **Deployment:** Plan to run live paper trading daily?
   - If yes, need robust data pipeline + automation
   - If no, can focus more on research + validation

---

## 📚 Resources & References

### Code Architecture Patterns
- **Portfolio Manager:** See `portfolio_manager.py` stub (to be created)
- **Multi-Asset Backtest:** Extend `run_walk_forward()` to accept dict of signals
- **Risk Budgeting:** Allocate volatility, not capital

### Statistical Testing
- **Permutation Tests:** `scipy.stats.permutation_test()`
- **HMM:** `hmmlearn` library for regime detection
- **Shrinkage Estimators:** `sklearn.covariance.LedoitWolf`

### Data Sources (if needed later)
- Futures data: Quandl, Interactive Brokers, or CSI Data
- Intraday data: Polygon.io, Alpha Vantage
- Alternative data: Sentiment, options flow (advanced)

---

## ✅ Definition of Done

The system is **production-ready** when:

1. ✅ Can run 5+ strategies across 5+ assets simultaneously
2. ✅ Walk-forward Sharpe > 1.0 with realistic costs
3. ✅ Permutation tests show p < 0.05 (statistically significant)
4. ✅ All components modular and independently testable
5. ✅ Full documentation with architecture diagrams
6. ✅ Can run daily paper trades with 1-command execution
7. ✅ Final report generated with all validation tests

---

**Last Updated:** November 22, 2025  
**Status:** Week 2 in progress (optimization framework built, per-fold optimization pending)
