# Multi-Strategy Portfolio Architecture & Live Dashboard Planning

## Document Purpose
This document outlines the architectural design for extending the current single-strategy paper trading system into a production-grade multi-strategy portfolio management system with integrated risk management and live web-based monitoring.

---

## Table of Contents
1. [Current State Analysis](#current-state-analysis)
2. [Multi-Strategy Portfolio Manager](#multi-strategy-portfolio-manager)
3. [Risk Manager Evolution](#risk-manager-evolution)
4. [Dashboard & Reporting System](#dashboard--reporting-system)
5. [Live Web Interface](#live-web-interface)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Technical Considerations](#technical-considerations)

---

## Current State Analysis

### Existing Architecture
- **PaperTradingEngine**: Manages single strategy with state persistence
- **PortfolioManager**: Runs backtests for single strategy
- **RiskManager**: Position sizing, leverage limits, violation tracking for single strategy
- **RiskDashboard**: Static HTML risk visualizations
- **Reporter**: Static HTML performance reports
- **Separate HTML files**: User must open multiple files

### Limitations
- No multi-strategy coordination
- No portfolio-level aggregation across strategies
- Risk limits applied per-strategy, not portfolio-wide
- No live updating - requires re-running code
- Fragmented dashboard experience
- No strategy attribution or correlation analysis

---

## Multi-Strategy Portfolio Manager

### Architecture Overview

#### 1. Strategy Container Structure
```
MultiStrategyPortfolio
├── Strategy 1: momentum_120
│   ├── PaperTradingEngine instance
│   ├── Signal generator (MomentumSignalV2)
│   ├── Independent equity curve
│   ├── Independent trades log
│   └── Strategy-specific state file
├── Strategy 2: mean_reversion_20
│   └── [same structure]
├── Strategy 3: trend_following_50
│   └── [same structure]
└── Portfolio Aggregator
    ├── Combined equity curve
    ├── Capital allocation weights
    ├── Rebalancing logic
    └── Portfolio-level state
```

#### 2. Capital Allocation Methods

**Static Allocation**
- Fixed percentage per strategy (e.g., 40% momentum, 30% mean reversion, 30% trend)
- Simplest to implement and understand
- Good for initial testing phase

**Dynamic Allocation (Risk Parity)**
- Weight inversely to volatility (lower vol = higher weight)
- Maintain equal risk contribution per strategy
- Rebalance monthly/quarterly based on realized vol

**Dynamic Allocation (Performance-Based)**
- Weight by recent Sharpe ratio or Sortino ratio
- Higher performing strategies get more capital
- Prevent over-allocation to lucky strategies (use minimum allocation floors)

**Dynamic Allocation (Kelly Criterion)**
- Optimal bet sizing based on edge and volatility
- Most aggressive but mathematically optimal
- Requires accurate win rate and payoff ratio estimates

**Hierarchical Risk Budgeting**
- Top-down capital allocation
- Portfolio risk budget → Strategy risk budgets → Position sizing
- Most sophisticated institutional approach

#### 3. Portfolio-Level Operations

**Initialization**
- Load all strategy configurations from config file
- Initialize each strategy's PaperTradingEngine
- Set initial capital allocation weights
- Create portfolio-level state persistence
- Validate total allocation = 100% (or with leverage target)

**Daily Update Workflow**
1. Fetch latest market data (shared across strategies)
2. Update each strategy independently (generate signals, execute trades)
3. Collect strategy-level equity values
4. Calculate portfolio-level metrics (total equity, cash, positions)
5. Check portfolio-level risk limits
6. Trigger rebalancing if allocation drift exceeds threshold
7. Save portfolio state + individual strategy states

**Rebalancing Logic**
- Trigger conditions:
  - Allocation drift > threshold (e.g., 5%)
  - Scheduled periodic rebalancing (monthly)
  - Manual override
  - Portfolio-level risk breach
- Rebalancing methods:
  - Cash injection/withdrawal to strategies
  - Proportional scaling of existing positions
  - Complete liquidation and reallocation (most disruptive)

**Portfolio Aggregation**
- Combined equity curve: Sum of all strategy equity values
- Combined positions: Merge position dictionaries (handle overlapping tickers)
- Combined trades: Concatenate with strategy identifier column
- Combined P&L: Strategy-level P&L plus allocation effects

#### 4. Strategy Correlation & Diversification

**Correlation Matrix**
- Calculate pairwise strategy return correlations
- Track correlation stability over time
- Alert when correlation regime shifts (strategies become too correlated)

**Diversification Benefits**
- Calculate portfolio volatility vs. sum of strategy volatilities
- Measure diversification ratio (portfolio vol / weighted avg vol)
- Track how much risk is reduced through diversification

**Strategy Attribution**
- Decompose portfolio returns into strategy contributions
- Identify which strategies drive performance/losses
- Calculate marginal contribution to risk (MCR) per strategy
- Brinson attribution: allocation effect vs. selection effect

---

## Risk Manager Evolution

### Current Risk Manager Limitations
- Operates on single strategy view
- No awareness of portfolio-level exposures
- Cannot manage cross-strategy correlations
- No netting of offsetting positions across strategies

### Portfolio-Level Risk Manager Design

#### 1. Hierarchical Risk Structure

**Portfolio Level (Top)**
- Maximum total portfolio leverage (e.g., 2.0x)
- Maximum portfolio drawdown stop (e.g., -15%)
- Maximum portfolio volatility target (e.g., 20% annualized)
- Sector/asset class concentration limits
- Maximum correlation between strategies
- Minimum diversification ratio

**Strategy Level (Middle)**
- Per-strategy leverage limits (e.g., 1.5x per strategy)
- Per-strategy drawdown stops (e.g., -10%)
- Per-strategy volatility targets
- Strategy-level position count limits

**Position Level (Bottom)**
- Maximum position size per ticker (e.g., 5% of strategy capital)
- Maximum position size per ticker portfolio-wide (aggregate across strategies)
- Stop-loss levels per position

#### 2. Cross-Strategy Risk Aggregation

**Position Netting**
- Strategy A: Long 10 ES contracts
- Strategy B: Short 5 ES contracts
- Net exposure: Long 5 ES contracts
- Risk calculations based on net exposure

**Correlation Adjustment**
- Two strategies holding same ticker but different signals
- Calculate correlation-adjusted risk (not just sum of individual risks)
- Value-at-Risk (VaR) aggregation using correlation matrix

**Concentration Risk**
- Track total exposure per ticker across all strategies
- Enforce portfolio-wide concentration limits
- Prevent multiple strategies from overloading same position

#### 3. Risk Metrics Calculation

**Strategy-Level Metrics** (per strategy)
- Leverage (strategy capital utilization)
- Number of positions
- Maximum position weight
- Strategy volatility (rolling 30/60/90 day)
- Strategy drawdown
- Sharpe ratio
- Win rate, avg trade P&L

**Portfolio-Level Metrics** (aggregated)
- Total portfolio leverage
- Net position count (after netting)
- Gross position count (sum of all positions)
- Portfolio volatility (correlation-adjusted)
- Portfolio drawdown
- Portfolio Sharpe ratio
- Strategy correlation matrix
- Diversification ratio
- Marginal risk contribution per strategy
- Value-at-Risk (VaR) - 95%, 99%
- Expected Shortfall (CVaR)
- Maximum adverse excursion (MAE)

#### 4. Violation Management

**Violation Hierarchy**
- **Critical**: Portfolio drawdown breach → Liquidate all strategies
- **High**: Portfolio leverage breach → Reduce positions proportionally
- **Medium**: Strategy drawdown breach → Pause that strategy
- **Low**: Position size breach → Reduce specific position

**Violation Actions**
- Log violation with severity, timestamp, affected strategy/position
- Trigger automated response based on severity
- Send alerts (email, SMS, webhook)
- Update dashboard with violation alerts
- Prevent new trades until resolved

**Violation Recovery**
- Define recovery conditions (e.g., drawdown recovers to -5%)
- Gradual re-entry (don't immediately go back to full risk)
- Post-violation review and parameter adjustment

---

## Dashboard & Reporting System

### Unified Multi-Page Dashboard Design

#### Page 1: Portfolio Overview (Landing Page)

**Key Metrics Panel**
- Total portfolio value (large, prominent)
- Today's P&L (absolute and %)
- Total return since inception
- Current leverage ratio
- Current drawdown
- Number of active strategies
- Number of open positions (gross and net)

**Portfolio Equity Curve** (main chart)
- Combined portfolio equity over time
- Overlay individual strategy equity curves (thin lines, different colors)
- Toggle visibility per strategy
- Benchmark comparison (S&P 500)
- Highlight rebalancing events

**Strategy Performance Table**
- Columns: Strategy Name, Capital Allocated, Current Value, Return, Sharpe, Drawdown, # Positions, Status
- Sortable by any column
- Color-coded performance (green/red)
- Click row to drill into strategy details

**Current Positions Summary**
- Top 10 positions by portfolio weight
- Show ticker, net exposure, gross exposure, P&L
- Pie chart of sector/asset class allocation

**Recent Activity Feed**
- Last 20 trades across all strategies
- Violations and alerts
- Rebalancing events
- Strategy status changes

#### Page 2: Risk Dashboard

**Portfolio Risk Metrics**
- Leverage gauge (current vs. limit)
- Volatility gauge (current vs. target)
- Drawdown chart (historical drawdown curve)
- VaR/CVaR metrics with confidence intervals

**Strategy Correlation Heatmap**
- Interactive heatmap of strategy return correlations
- Click cell to see scatter plot of returns
- Alert if correlations exceed threshold

**Position Concentration**
- Bar chart: Top 10 positions by portfolio weight
- Stacked bar showing which strategies hold each position
- Concentration limit lines

**Risk Decomposition**
- Pie chart: Risk contribution per strategy (MCR)
- Show which strategies contribute most to portfolio vol
- Highlight diversification benefits

**Violation Log**
- Table of all violations (historical)
- Filter by severity, date range, strategy
- Show violation type, action taken, resolution

**Stress Testing** (future enhancement)
- Scenario analysis: What if X market moves Y%
- Historical stress tests (2008, 2020, etc.)
- Strategy-specific shocks

#### Page 3: Performance Analytics

**Returns Analysis**
- Cumulative returns chart (portfolio + strategies)
- Daily/monthly/yearly returns histogram
- Returns distribution with normal overlay
- Drawdown periods highlighted

**Monthly Returns Heatmap**
- Rows = years, columns = months
- Portfolio-level heatmap
- Hover to see strategy-level breakdown

**Trade Analysis**
- Trade P&L distribution (histogram)
- Win rate by strategy
- Average winner vs. average loser
- Hold time distribution
- Trade frequency over time

**Strategy Attribution**
- Stacked area chart: Cumulative return contribution per strategy
- Bar chart: Period return contribution
- Table: Detailed attribution metrics

**Rolling Metrics**
- Rolling Sharpe ratio (30/60/90 day windows)
- Rolling volatility
- Rolling beta to benchmark
- Rolling correlation between strategies

#### Page 4: Strategy Deep Dive

**Strategy Selector Dropdown**
- Choose which strategy to analyze in detail

**Strategy-Specific Metrics**
- All the metrics currently in single-strategy dashboard
- Equity curve with trades overlaid
- Signal vs. entry/exit visualization
- Strategy-specific violations

**Strategy Positions**
- Detailed position table
- Entry prices, current prices, P&L
- Hold duration
- Position sizing rationale

**Strategy Trades**
- Complete trade history
- Trade analytics (similar to current Reporter)
- Export to CSV

#### Page 5: Configuration & Controls

**Strategy Management**
- Add/remove strategies
- Modify allocation weights
- Pause/resume strategies
- Change risk parameters

**Rebalancing Controls**
- Trigger manual rebalance
- View rebalancing history
- Configure automatic rebalancing rules

**Alert Configuration**
- Set custom alert thresholds
- Email/SMS notification settings
- Webhook URLs for integrations

**Data Export**
- Download portfolio equity curve
- Download all trades
- Download risk metrics
- Export to Excel/CSV

---

## Live Web Interface

### Technology Stack Options

#### Option 1: Static HTML + Auto-Refresh (Simplest)
**Technology**: Generate HTML files on every code run
**Access**: File system or simple HTTP server
**Pros**:
- Zero server maintenance
- Works offline
- Very fast page loads
**Cons**:
- Must manually re-run code to update
- No real-time updates
- No user interactivity beyond clicks

**Implementation**:
- Current approach, but unified multi-page HTML
- Use HTML nav bar to link pages
- Add `<meta http-equiv="refresh" content="60">` for auto-reload
- Simple Python HTTP server: `python -m http.server 8000`

#### Option 2: Flask + Periodic Background Updates (Recommended)
**Technology**: Flask web framework + APScheduler
**Access**: `http://localhost:5000` or deployed URL
**Pros**:
- Real web server with routing
- Scheduled background updates (every minute/hour)
- Can trigger manual updates via button click
- Database for historical data storage
- User authentication possible
**Cons**:
- Requires server process running
- More complex to deploy
- Need to handle state/session management

**Architecture**:
```
Flask App
├── Background Scheduler (APScheduler)
│   └── Runs update_portfolio() every X minutes
├── Routes
│   ├── /                    → Portfolio Overview
│   ├── /risk                → Risk Dashboard
│   ├── /performance         → Performance Analytics
│   ├── /strategy/<name>     → Strategy Deep Dive
│   ├── /config              → Configuration Panel
│   └── /api/update          → Manual update trigger
├── Templates (Jinja2)
│   └── HTML pages with embedded Plotly charts
├── Database (SQLite/PostgreSQL)
│   ├── equity_history
│   ├── trades
│   ├── risk_metrics
│   └── strategy_states
└── Static Assets
    ├── CSS (Bootstrap)
    └── JS (Plotly, jQuery)
```

**Background Update Process**:
1. APScheduler triggers `update_portfolio()` every N minutes
2. Function loads all strategy engines from disk
3. Fetches latest market data
4. Updates each strategy (signals + trades)
5. Saves new state to database
6. Caches rendered charts for fast page loads
7. Logs completion timestamp

**User Triggers Manual Update**:
1. Click "Update Now" button on web interface
2. POST request to `/api/update`
3. Triggers same update process
4. Returns JSON status
5. JavaScript auto-refreshes page

#### Option 3: Dash by Plotly (Most Interactive)
**Technology**: Dash framework (built on Flask)
**Access**: `http://localhost:8050`
**Pros**:
- Beautiful interactive Plotly charts
- Callbacks for real-time interactivity
- Python-only (no JavaScript needed)
- Great for data science dashboards
**Cons**:
- Steeper learning curve
- More opinionated framework
- Can be slow with many components

**Use Case**: Best for exploratory analysis and internal tools

#### Option 4: Streamlit (Fastest Prototyping)
**Technology**: Streamlit framework
**Access**: `http://localhost:8501`
**Pros**:
- Incredibly fast to build
- Automatic reruns on code change
- Beautiful default styling
- Trivial to deploy (Streamlit Cloud)
**Cons**:
- Full page reload on interaction (can be slow)
- Less control over layout
- Not ideal for complex multi-page apps

**Use Case**: Quick prototypes and simple dashboards

#### Option 5: Next.js + FastAPI (Production Grade)
**Technology**: React frontend + Python backend API
**Access**: Full web application
**Pros**:
- Professional, scalable architecture
- Real-time WebSocket updates
- Mobile responsive
- Can deploy to cloud (AWS, Vercel)
**Cons**:
- Requires JavaScript/TypeScript knowledge
- Significantly more complex
- Overkill for personal project

**Use Case**: If building commercial product or team platform

### Recommended Approach: Flask + Background Updates

**Why Flask?**
1. **Balance**: More powerful than static HTML, simpler than React
2. **Python-native**: No need to learn JavaScript
3. **Extensible**: Easy to add features incrementally
4. **Deployable**: Can run on local machine or cloud
5. **Community**: Huge ecosystem of plugins and examples

**Deployment Options**:

**Local Development**:
```bash
python run_dashboard.py
# Dashboard available at http://localhost:5000
# Automatically updates every 5 minutes
```

**Raspberry Pi / Home Server**:
- Run Flask app as systemd service
- Access from any device on local network: `http://192.168.1.100:5000`
- No internet required, data stays private

**Cloud Deployment (AWS/Heroku/DigitalOcean)**:
- Deploy Flask app to cloud server
- Access from anywhere: `https://yourdashboard.com`
- Set up SSL certificate for security
- Use PostgreSQL for database (more robust than SQLite)
- Add authentication (username/password)

**Docker Container**:
- Package entire app in Docker image
- Portable across any system
- Easy to update and rollback versions
- Include database, scheduler, all dependencies

### Real-Time Updates (Advanced)

**WebSocket Integration**:
- Use Flask-SocketIO for real-time communication
- Server pushes updates to browser without page refresh
- Live P&L ticker, live position updates
- "Another trade just executed" notifications

**Implementation**:
- When background update completes, emit WebSocket event
- JavaScript listener receives event and updates DOM
- Only update changed elements (not full page reload)
- Feels like native app

---

## Implementation Roadmap

### Phase 1: Multi-Strategy Foundation (2-3 weeks)
**Goal**: Portfolio manager can run multiple strategies

**Tasks**:
1. Design `MultiStrategyPortfolio` class API
2. Implement strategy container (dict of engines)
3. Add capital allocation configuration (YAML/JSON)
4. Implement portfolio state persistence
5. Create portfolio aggregation methods (combine equity curves, trades)
6. Write unit tests for multi-strategy operations
7. Update notebook to test 2-3 simple strategies

**Deliverables**:
- `core/multi_strategy_portfolio.py` - Main portfolio manager
- `config/strategy_allocation.yaml` - Configuration file
- `tests/test_multi_strategy.py` - Test suite
- Working notebook demonstrating 2 strategies

**Success Criteria**:
- Can run 3 different momentum strategies (lookback 60/120/200)
- Portfolio equity = sum of strategy equities
- Can save/load portfolio state
- All tests pass

### Phase 2: Portfolio-Level Risk Management (2 weeks)
**Goal**: Risk manager operates at portfolio level

**Tasks**:
1. Design hierarchical risk structure (portfolio → strategy → position)
2. Implement portfolio-level risk metrics calculation
3. Add position netting logic (across strategies)
4. Implement correlation-adjusted risk calculations
5. Create portfolio-level violation detection
6. Add violation action framework (auto-response)
7. Update risk dashboard for portfolio view

**Deliverables**:
- `core/portfolio_risk_manager.py` - Enhanced risk manager
- Updated `core/risk_dashboard.py` - Multi-strategy visualizations
- Integration tests with multi-strategy portfolio
- Documentation for new risk metrics

**Success Criteria**:
- Can track portfolio leverage correctly
- Detects when 2 strategies hold same position (netting)
- Violation hierarchy works (critical → high → medium → low)
- Risk dashboard shows portfolio + strategy levels

### Phase 3: Unified Dashboard Generation (1-2 weeks)
**Goal**: Single multi-page HTML dashboard

**Tasks**:
1. Design unified dashboard navigation structure
2. Refactor `RiskDashboard` and `Reporter` into `UnifiedDashboard`
3. Implement 5-page layout (overview, risk, performance, strategy, config)
4. Add JavaScript for page navigation (no separate files)
5. Create portfolio overview page with strategy table
6. Add strategy attribution charts
7. Implement strategy drill-down page

**Deliverables**:
- `core/unified_dashboard.py` - New dashboard generator
- Single HTML file with tab/page navigation
- All current visualizations migrated to new format
- Example notebooks generating unified dashboard

**Success Criteria**:
- One HTML file, multiple pages/tabs
- Can navigate between pages without opening new files
- Shows portfolio + individual strategy views
- All Plotly charts work correctly

### Phase 4: Flask Web Application (2-3 weeks)
**Goal**: Live updating web dashboard

**Tasks**:
1. Set up Flask project structure
2. Create database schema (SQLite initially)
3. Implement API routes for all dashboard pages
4. Create Jinja2 templates for each page
5. Add APScheduler for background updates
6. Implement manual update button
7. Add configuration page for strategy management
8. Create `/api/status` endpoint for health checks
9. Add basic logging and error handling
10. Write deployment documentation

**Deliverables**:
- `dashboard/` directory with Flask app
  - `app.py` - Main Flask application
  - `scheduler.py` - Background update jobs
  - `models.py` - Database models
  - `routes.py` - Web routes
  - `templates/` - HTML templates
  - `static/` - CSS, JS, images
- `run_dashboard.py` - Launch script
- Docker configuration (optional)
- README for deployment

**Success Criteria**:
- Can run Flask app locally: `python run_dashboard.py`
- Dashboard auto-updates every 5 minutes
- "Update Now" button triggers immediate update
- Can view on any device on local network
- Database stores historical data correctly
- Survives restarts (loads from database)

### Phase 5: Advanced Features (Ongoing)
**Goal**: Polish and production hardening

**Tasks**:
- Add user authentication (username/password)
- Implement strategy correlation regime detection
- Add stress testing / scenario analysis
- Create email/SMS alert system
- Add webhook integrations (Discord, Slack, PagerDuty)
- Implement trade approval workflow (manual override)
- Add position-level annotations (notes, tags)
- Create strategy comparison tool (side-by-side backtests)
- Add Monte Carlo simulation for forward projections
- Implement performance attribution decomposition

---

## Technical Considerations

### State Management

**Current Approach**: Pickle files per strategy
- Works well for single strategy
- Simple serialization

**Multi-Strategy Approach**:
- **Option A**: One pickle file per strategy + one portfolio pickle
  - Pros: Modular, strategies independent
  - Cons: Must sync multiple files, risk of inconsistency
- **Option B**: Single pickle file for entire portfolio
  - Pros: Atomic saves, guaranteed consistency
  - Cons: Large file size, must reload everything
- **Option C**: Database (SQLite or PostgreSQL)
  - Pros: Query historical data, no file corruption, concurrent access
  - Cons: More complex, need migration scripts
  - **Recommended for production**

**Database Schema Design**:
```
strategies
- strategy_id (PK)
- name
- config (JSON)
- status (active/paused)
- created_at

portfolio_equity
- timestamp (PK)
- total_value
- cash
- leverage
- drawdown
- volatility

strategy_equity
- timestamp (PK)
- strategy_id (FK)
- equity_value
- cash
- num_positions

positions
- position_id (PK)
- strategy_id (FK)
- ticker
- shares
- entry_price
- entry_date
- current_price
- unrealized_pnl

trades
- trade_id (PK)
- strategy_id (FK)
- timestamp
- ticker
- type (buy/sell)
- shares
- price
- value
- pnl
- transaction_cost

risk_violations
- violation_id (PK)
- timestamp
- severity (critical/high/medium/low)
- type
- affected_strategy_id (FK, nullable)
- description
- action_taken
- resolved_at
```

### Performance Optimization

**Problem**: Multiple strategies = more computation
- 10 strategies × 5000 bars each = 50,000 calculations
- Dashboard generation can be slow

**Solutions**:
1. **Incremental Updates**: Only recalculate new data, not entire history
2. **Caching**: Cache rendered charts, only regenerate on data change
3. **Parallel Processing**: Update strategies in parallel (multiprocessing)
4. **Lazy Loading**: Dashboard pages load data on-demand (AJAX)
5. **Data Aggregation**: Pre-calculate daily/weekly/monthly summaries

**Memory Management**:
- Don't load all historical data into RAM
- Use database queries to fetch only required date ranges
- Stream large datasets (pandas chunking)

### Code Organization

**Suggested Project Structure**:
```
QuantTrading/
├── core/
│   ├── multi_strategy_portfolio.py      # NEW: Portfolio manager
│   ├── portfolio_risk_manager.py        # NEW: Enhanced risk manager
│   ├── unified_dashboard.py             # NEW: Dashboard generator
│   ├── paper_trading_engine.py          # Existing
│   ├── portfolio_manager.py             # Existing
│   └── ...
├── dashboard/                            # NEW: Flask web app
│   ├── app.py
│   ├── routes.py
│   ├── scheduler.py
│   ├── models.py
│   ├── database.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── overview.html
│   │   ├── risk.html
│   │   ├── performance.html
│   │   ├── strategy.html
│   │   └── config.html
│   └── static/
│       ├── css/
│       ├── js/
│       └── images/
├── config/
│   ├── strategy_allocation.yaml         # NEW: Multi-strategy config
│   └── risk_limits.yaml                 # NEW: Portfolio risk config
├── data/
│   ├── portfolio_state.db               # NEW: SQLite database
│   └── strategy_states/                 # Individual strategy pickles
├── notebooks/
│   ├── testing/
│   │   └── 11_multi_strategy_portfolio.ipynb  # NEW
│   └── ...
├── tests/
│   ├── test_multi_strategy_portfolio.py # NEW
│   ├── test_portfolio_risk_manager.py   # NEW
│   └── ...
├── docs/
│   ├── MULTI_STRATEGY_ARCHITECTURE.md   # This document
│   └── ...
├── run_dashboard.py                      # NEW: Launch web dashboard
├── requirements.txt                      # Update with Flask, etc.
└── README.md
```

### Testing Strategy

**Unit Tests**:
- Test each strategy runs independently
- Test portfolio aggregation math (equity curves sum correctly)
- Test capital allocation logic (weights sum to 100%)
- Test position netting across strategies
- Test risk limit violations trigger correctly

**Integration Tests**:
- Test full multi-strategy portfolio workflow
- Test state save/load with multiple strategies
- Test dashboard generation with real data
- Test Flask app endpoints return correct data

**Simulation Tests**:
- Run multi-strategy portfolio on historical data
- Verify portfolio return = sum of weighted strategy returns
- Test correlation calculations are correct
- Verify rebalancing logic doesn't create value/lose money

### Configuration Management

**Strategy Configuration File** (`config/strategy_allocation.yaml`):
```yaml
portfolio:
  name: "Multi-Strategy Momentum Portfolio"
  initial_capital: 100000
  rebalancing:
    method: "periodic"  # periodic, threshold, manual
    frequency: "monthly"
    threshold: 0.05  # 5% drift triggers rebalance

strategies:
  - name: "momentum_60"
    allocation: 0.30
    signal_type: "MomentumSignalV2"
    parameters:
      lookback: 60
      sma_filter: 200
    risk_config:
      max_position_size: 0.35
      max_leverage: 1.0
      volatility_target: 0.24
    status: "active"
  
  - name: "momentum_120"
    allocation: 0.40
    signal_type: "MomentumSignalV2"
    parameters:
      lookback: 120
      sma_filter: 200
    risk_config:
      max_position_size: 0.35
      max_leverage: 1.0
      volatility_target: 0.24
    status: "active"
  
  - name: "momentum_200"
    allocation: 0.30
    signal_type: "MomentumSignalV2"
    parameters:
      lookback: 200
      sma_filter: 200
    risk_config:
      max_position_size: 0.35
      max_leverage: 1.0
      volatility_target: 0.24
    status: "paused"  # Can pause strategies
```

**Portfolio Risk Configuration** (`config/portfolio_risk_limits.yaml`):
```yaml
portfolio_limits:
  max_leverage: 2.0          # Portfolio-wide
  max_drawdown: -0.15        # -15% from peak
  max_volatility: 0.25       # 25% annualized
  min_diversification_ratio: 1.2  # Must see diversification benefit
  
  position_limits:
    max_weight_per_ticker: 0.10      # 10% portfolio-wide
    max_concentration_top5: 0.40     # Top 5 positions <= 40%
  
  correlation_limits:
    max_strategy_correlation: 0.80   # Strategies shouldn't be too correlated
    alert_threshold: 0.70            # Alert if correlation rising
  
  violation_actions:
    critical:  "liquidate_all"       # Portfolio DD breach
    high:      "reduce_positions"    # Portfolio leverage breach
    medium:    "pause_strategy"      # Strategy DD breach
    low:       "alert_only"          # Position size breach

alert_settings:
  email:
    enabled: true
    recipients: ["your_email@example.com"]
    smtp_server: "smtp.gmail.com"
  
  slack:
    enabled: false
    webhook_url: "https://hooks.slack.com/..."
  
  sms:
    enabled: false
    twilio_sid: "..."
    twilio_token: "..."
    phone_number: "+1234567890"
```

### Security Considerations

**Local Deployment**:
- No authentication needed (only you have access)
- Firewall rules to prevent external access

**Network Deployment**:
- Add username/password authentication (Flask-Login)
- Use HTTPS (SSL certificate)
- Rate limiting to prevent brute force
- Session management with secure cookies

**API Keys / Credentials**:
- Never commit API keys to Git
- Use environment variables: `os.getenv('YFINANCE_API_KEY')`
- Consider using `.env` file + python-dotenv library

**Data Protection**:
- Encrypt sensitive data in database (strategy parameters, P&L)
- Regular backups of database
- Access logs (who viewed what, when)

---

## Conclusion

This architecture provides a clear path from the current single-strategy system to a production-grade multi-strategy portfolio management platform. The phased approach allows for incremental development while maintaining a working system at each stage.

**Key Design Principles**:
1. **Modularity**: Strategies are independent but coordinated
2. **Scalability**: Easy to add new strategies without refactoring
3. **Hierarchy**: Clear separation of portfolio/strategy/position concerns
4. **Observability**: Rich dashboards and logging at every level
5. **Safety**: Multiple layers of risk management and violation handling
6. **Usability**: Live web interface for seamless monitoring

**Next Steps**:
1. Review and refine this architecture document
2. Learn current codebase through notebooks (as planned)
3. Implement walk-forward validation
4. Begin Phase 1 (Multi-Strategy Foundation) when ready

This design will support professional-grade portfolio management suitable for personal trading, small fund management, or institutional research applications.
