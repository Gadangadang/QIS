# Commodities & Multi-Asset Class Expansion Plan

**Date:** December 1, 2025  
**Objective:** Expand the QuantTrading system to handle energy commodities (oil, gas, electricity), metals, and agricultural futures with the same institutional rigor as equity futures, while maintaining true multi-asset flexibility.

---

## 🎯 Vision Statement

Build a **truly agnostic multi-asset quantitative trading system** where:
- Any asset class (equities, commodities, FX, fixed income) can be traded with equal sophistication
- Commodity-specific features (seasonality, carry, storage costs, term structure) are first-class citizens
- Futures rollover is automatic and backtestable
- Position sizing respects contract specifications and margin requirements
- Signal generation adapts to asset characteristics (trend-following for oil, mean-reversion for gas spreads)

**End Goal:** A system where you can trade crude oil with the same ease and rigor as ES futures, seamlessly switching between intraday electricity strategies and monthly agricultural spread trades.

---

## 📊 Current State Assessment

### ✅ What We Have
**Strong Foundation:**
- Multi-asset loader supporting ES, NQ, GC
- Portfolio manager with pluggable position sizers
- Risk controls (stop-loss, take-profit, position limits)
- Walk-forward validation framework
- Multiple signal generators (momentum, mean reversion, trend following, ensemble)
- Reporter and risk dashboard infrastructure
- Vectorized backtesting (no loops)

**Capabilities:**
- Can handle multiple equity futures simultaneously
- Capital allocation across strategies
- Transaction cost modeling (3 bps + 2 bps slippage)
- Risk budgeting and correlation tracking

### ❌ What's Missing for Commodities

**Critical Gaps:**

1. **Futures Rollover Infrastructure**
   - No expiration date tracking
   - No automatic contract switching
   - No rollover cost modeling
   - No continuous vs. nearest contract logic

2. **Commodity-Specific Data Models**
   - No term structure support (front month vs. back months)
   - No storage cost modeling
   - No seasonality patterns
   - No carry calculations

3. **Asset Classification System**
   - Hard-coded asset list in `MultiAssetLoader`
   - No differentiation between asset types (futures vs. stocks vs. ETFs)
   - No contract specifications (multipliers, tick sizes)
   - No margin requirement tracking

4. **Commodity-Specific Signals**
   - No calendar spread strategies (Dec CL - Jan CL)
   - No carry-based signals (contango vs. backwardation)
   - No seasonal patterns (winter gas, summer driving oil)
   - No inter-commodity spreads (crack spread, spark spread)

5. **Intraday Capability**
   - Only daily data support
   - No intraday electricity price handling
   - No hourly/minute bar support
   - No time-of-day filters

6. **Execution Realism**
   - No rollover slippage
   - No storage/carry costs
   - No settlement vs. trading price distinction
   - No physical delivery considerations

---

## 🏗️ Architecture Design

### Core Components to Build

#### 1. **Asset Registry & Metadata System** ⭐ FOUNDATION

**Purpose:** Single source of truth for ALL assets

```python
# Core enums and data structures
- AssetType: FUTURES, STOCK, ETF, INDEX, FOREX
- AssetClass: EQUITY, COMMODITY_ENERGY, COMMODITY_METAL, COMMODITY_AGRICULTURE, FIXED_INCOME, CURRENCY, VOLATILITY
- AssetMetadata: ticker, name, type, class, contract_multiplier, tick_size, expiration_cycle, margin_pct, seasonality info

# Registry
ASSET_REGISTRY: Dict[str, AssetMetadata]
  - Extensible: add new assets without code changes
  - Validated: type checking on all fields
  - Queryable: filter by asset_class, requires_rollover, etc.
```

**Benefits:**
- One place to define all asset characteristics
- Automatic validation (can't create position larger than multiplier allows)
- Easy to add new asset classes (just add to registry)
- Supports filtering (e.g., "all energy commodities requiring rollover")

**Integration Points:**
- `MultiAssetLoader`: Use registry to determine yfinance symbols
- `PortfolioManager`: Use contract_multiplier for position sizing
- `RiskManager`: Use typical_margin_pct for margin calculations
- `SignalGenerators`: Query has_seasonality for strategy selection

---

#### 2. **Futures Rollover Handler** ⭐ CRITICAL

**Purpose:** Automatic contract expiration management with backtestable history

**Key Features:**

```python
class FuturesRolloverHandler:
    """
    Handles futures contract rollovers with proper adjustments.
    
    Features:
    - Expiration calendar (CME, ICE, NYMEX schedules)
    - Rollover rules (5 days before expiry, volume-based, etc.)
    - Price adjustments (Panama vs. ratio method)
    - Rollover cost tracking
    """
    
    def get_active_contract(self, ticker: str, date: pd.Timestamp) -> str:
        """Return active contract symbol for given date (e.g., 'CLZ24')"""
        
    def create_continuous_series(self, ticker: str, method: str = 'panama') -> pd.DataFrame:
        """Build continuous price series from individual contracts"""
        
    def get_rollover_dates(self, ticker: str, start: str, end: str) -> List[pd.Timestamp]:
        """Return all rollover dates in period"""
        
    def calculate_rollover_cost(self, front_price: float, back_price: float) -> float:
        """Calculate cost of rolling from front to back month"""
```

**Rollover Methods:**
- **Panama Method:** Adjust historical prices by roll difference (most common)
- **Ratio Method:** Adjust by roll ratio (better for percentage-based strategies)
- **No Adjustment:** Keep raw prices (for spread trading)

**Expiration Calendars:**
- Energy (CL, NG): Monthly, 3rd business day before 25th
- Metals (GC, HG): Monthly, 3rd last business day
- Equity Indices (ES, NQ): Quarterly, 3rd Friday (CME schedule)
- Custom: User-defined schedules

**Rollover Triggers:**
- **Days-based:** Roll X days before expiration (default 5-7 days)
- **Volume-based:** Roll when back month volume > front month
- **OI-based:** Roll when back month open interest > front month
- **Fixed date:** Always roll on specific calendar day

**Backtest Integration:**
```python
# In backtester
rollover_handler = FuturesRolloverHandler()

for date in trading_dates:
    active_contract = rollover_handler.get_active_contract('CL', date)
    
    if rollover_handler.is_rollover_date('CL', date):
        # Execute roll
        front_price = prices[active_contract]
        back_contract = rollover_handler.get_next_contract('CL', date)
        back_price = prices[back_contract]
        
        # Calculate rollover cost
        roll_cost = rollover_handler.calculate_rollover_cost(front_price, back_price)
        
        # Update position (close front, open back)
        portfolio.roll_position('CL', front_contract, back_contract, roll_cost)
```

**Data Structure:**
```python
# Store both individual contracts AND continuous series
data = {
    'CL': {
        'continuous': pd.DataFrame,  # Adjusted continuous prices
        'contracts': {
            'CLZ24': pd.DataFrame,   # Dec 2024 contract
            'CLF25': pd.DataFrame,   # Jan 2025 contract
            # ...
        },
        'rollover_dates': [pd.Timestamp('2024-11-15'), ...]
    }
}
```

---

#### 3. **Enhanced Multi-Asset Loader** 🔧 UPGRADE

**Extend existing `MultiAssetLoader` to support:**

```python
class MultiAssetLoader:
    """Enhanced with commodity and futures support"""
    
    def __init__(self, asset_registry: Dict[str, AssetMetadata], rollover_handler: FuturesRolloverHandler):
        self.registry = asset_registry
        self.rollover_handler = rollover_handler
    
    def load_assets(self, tickers: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Load data with automatic futures rollover handling.
        
        New kwargs:
        - use_continuous: bool (default True) - Use adjusted continuous series
        - rollover_method: 'panama' | 'ratio' | 'none'
        - load_term_structure: bool - Include multiple contract months
        """
    
    def load_term_structure(self, ticker: str, date: pd.Timestamp, num_contracts: int = 3) -> pd.DataFrame:
        """Load front month, 2nd month, 3rd month for spread strategies"""
    
    def load_intraday(self, ticker: str, start: str, end: str, interval: str = '1h') -> pd.DataFrame:
        """Load intraday data (for electricity, gas)"""
```

**Data Sources:**
- **Historical:** CSV files + yfinance for recent data (current approach)
- **Intraday:** Polygon.io, Alpha Vantage, or Interactive Brokers API
- **Futures contracts:** Quandl, CSI Data, or IB historical data
- **Electricity:** EIA API (Energy Information Administration - free!)

**File Structure:**
```
Dataset/
  equity_futures/
    es_continuous_2000_2025.csv
    nq_continuous_2000_2025.csv
  energy/
    cl_continuous_1990_2025.csv
    cl_contracts/
      CLZ24.csv  # Individual contract data
      CLF25.csv
    ng_continuous_2000_2025.csv
    ng_intraday_2024_2025.csv  # Hourly data
  metals/
    gc_continuous_1990_2025.csv
    hg_continuous_1990_2025.csv
  electricity/
    pjm_day_ahead_hourly_2020_2025.csv  # PJM market
    ercot_real_time_5min_2024_2025.csv  # Texas market
```

---

#### 4. **Commodity-Specific Signal Generators** 🧠 NEW

**Purpose:** Signals that exploit commodity market structure

**A. Carry-Based Signals**
```python
class CarrySignal(SignalModel):
    """
    Trade based on term structure (contango vs. backwardation).
    
    Logic:
    - Backwardation (front > back): Go LONG (positive carry)
    - Contango (front < back): Go SHORT or avoid (negative carry)
    """
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Requires term_structure data (front_month, second_month prices)
        df['carry'] = (df['front_month'] - df['second_month']) / df['second_month']
        df['Signal'] = np.where(df['carry'] > self.threshold, 1, 0)
        return df
```

**B. Seasonal Signals**
```python
class SeasonalSignal(SignalModel):
    """
    Trade based on historical seasonal patterns.
    
    Examples:
    - Natural Gas: Long in summer (low prices), short in winter (high prices)
    - Crude Oil: Long before summer driving season
    - Agriculture: Plant/harvest cycles
    """
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add month, day-of-year features
        df['month'] = df['Date'].dt.month
        df['day_of_year'] = df['Date'].dt.dayofyear
        
        # Seasonal pattern (pre-calculated from historical data)
        seasonal_factors = self.load_seasonal_pattern(ticker)
        df['seasonal_signal'] = df['month'].map(seasonal_factors)
        
        df['Signal'] = np.where(df['seasonal_signal'] > 0, 1, -1)
        return df
```

**C. Calendar Spread Signals**
```python
class CalendarSpreadSignal(SignalModel):
    """
    Trade front month vs. back month spreads.
    
    Logic:
    - If spread too wide: Long front, short back (convergence trade)
    - If spread too narrow: Short front, long back
    """
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        # df must have 'front_month' and 'back_month' columns
        df['spread'] = df['front_month'] - df['back_month']
        df['spread_zscore'] = (df['spread'] - df['spread'].rolling(60).mean()) / df['spread'].rolling(60).std()
        
        df['Signal'] = 0
        df.loc[df['spread_zscore'] > 2, 'Signal'] = -1  # Spread too wide, short it
        df.loc[df['spread_zscore'] < -2, 'Signal'] = 1  # Spread too narrow, long it
        return df
```

**D. Crack Spread / Spark Spread**
```python
class CrackSpreadSignal(SignalModel):
    """
    Trade crude oil refining margins (CL vs. gasoline + heating oil).
    
    Crack Spread = (Gasoline + Heating Oil) / 2 - Crude Oil
    
    When crack spread is high: Refiners profitable, buy crude to refine
    When crack spread is low: Refiners losing money, sell crude
    """
```

**E. Storage Cost Arbitrage**
```python
class StorageArbitrageSignal(SignalModel):
    """
    If (back_month - front_month) > storage_cost:
        Buy front month, sell back month, earn storage yield
    """
```

**Signal Library Expansion:**
```python
# New signal categories
signals/
  commodities/
    __init__.py
    carry.py           # Carry-based strategies
    seasonal.py        # Seasonal patterns
    spreads.py         # Calendar spreads, inter-commodity spreads
    storage.py         # Storage arbitrage
    roll_yield.py      # Roll yield strategies
  energy_specific/
    electricity_intraday.py   # Time-of-day patterns
    gas_weather.py            # Weather-based gas trading
    oil_inventory.py          # Inventory report reactions
```

---

#### 5. **Intraday Data & Strategy Support** ⚡ NEW

**Purpose:** Enable high-frequency commodity trading (especially electricity)

**Challenges:**
- **Electricity:** No storage, must balance supply/demand in real-time
- **Prices:** Can spike 100x during peak demand (summer AC, winter heating)
- **Patterns:** Strong time-of-day effects (5pm peak vs. 3am off-peak)
- **Data volume:** Minute-level data = 252 days × 390 minutes = 98,000 rows/year

**Architecture:**

```python
class IntradayDataLoader:
    """Load and manage high-frequency data"""
    
    def load_hourly(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Load hourly bars (electricity, gas)"""
    
    def load_minute(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Load minute bars (for very short-term strategies)"""
    
    def resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate intraday to daily for compatibility with daily strategies"""

class IntradaySignal(SignalModel):
    """Base class for intraday strategies"""
    
    def generate(self, df: pd.DataFrame, time_window: str = '09:30-16:00') -> pd.DataFrame:
        """Generate signals only during specified time window"""
```

**Electricity-Specific Strategies:**

```python
class TimeOfDayMeanReversion(SignalModel):
    """
    Electricity prices mean-revert within day.
    
    Logic:
    - Morning spike above average: SHORT (revert by afternoon)
    - Morning dip below average: LONG (revert by afternoon)
    """

class PeakOffPeakSpread(SignalModel):
    """
    Trade peak hours (9am-9pm) vs. off-peak (9pm-9am).
    
    When spread > historical average: Sell peak, buy off-peak
    """

class WeatherDrivenElectricity(SignalModel):
    """
    Hot weather forecast → LONG electricity (AC demand)
    Cold weather forecast → LONG natural gas (heating demand)
    """
```

**Data Management:**
```python
# Store intraday separately from daily
data = {
    'NG': {
        'daily': pd.DataFrame,      # Standard daily OHLC
        'hourly': pd.DataFrame,     # 24 rows per day
        'metadata': AssetMetadata
    }
}

# Backtest both intraday and daily strategies
portfolio_manager.run_backtest(
    signals={
        'NG_daily': momentum_signal,     # Daily strategy
        'NG_intraday': time_of_day_signal  # Intraday strategy
    },
    prices={
        'NG_daily': daily_prices,
        'NG_intraday': hourly_prices
    }
)
```

---

#### 6. **Enhanced Position Sizing for Commodities** 💰 UPGRADE

**Current State:**
- `FixedFractionalSizer`: Fixed % of capital
- `KellySizer`: Kelly criterion
- `ATRSizer`: Volatility-based
- `VolatilityScaledSizer`: Inverse vol
- `RiskParitySizer`: Equal risk

**Commodity-Specific Needs:**

```python
class ContractMultiplierSizer:
    """
    Respect futures contract specifications.
    
    Example:
    - CL (crude oil): 1 contract = 1,000 barrels
    - NG (nat gas): 1 contract = 10,000 MMBtu
    - ES (S&P): 1 contract = 50 × index value
    
    If you want $10,000 exposure to CL at $70/barrel:
    - Naive: $10,000 / $70 = 142 barrels
    - Correct: 0.142 contracts (but must round to whole contracts)
    """
    
    def calculate_position(self, target_dollar_amount: float, asset: AssetMetadata, price: float) -> int:
        """Return number of contracts (integer)"""
        notional_per_contract = price * asset.contract_multiplier
        contracts = target_dollar_amount / notional_per_contract
        return round(contracts)  # Must be whole number

class MarginAwareSizer:
    """
    Don't over-leverage based on margin requirements.
    
    Example:
    - Capital: $100,000
    - CL margin: 8% of contract value
    - CL price: $70, multiplier: 1,000
    - Contract value: $70,000
    - Required margin: $5,600
    - Max contracts: $100,000 / $5,600 = 17 contracts
    """
```

**Integration with Portfolio Manager:**
```python
# When calculating position size
metadata = ASSET_REGISTRY[ticker]

if metadata.asset_type == AssetType.FUTURES:
    # Use contract-aware sizing
    sizer = ContractMultiplierSizer(metadata)
    position = sizer.calculate_position(target_dollars, metadata, current_price)
else:
    # Use standard sizing (for stocks/ETFs)
    position = target_dollars / current_price
```

---

#### 7. **Seasonality Analysis Framework** 🗓️ NEW

**Purpose:** Quantify and exploit seasonal patterns in commodities

**Features:**

```python
class SeasonalityAnalyzer:
    """
    Calculate and visualize seasonal patterns.
    """
    
    def calculate_seasonal_pattern(self, ticker: str, years: int = 10) -> pd.Series:
        """
        Return average return by month/day-of-year.
        
        Output: Series with 365 values (one per day-of-year)
        """
    
    def test_seasonality_significance(self, ticker: str) -> Dict:
        """
        Statistical test: Is seasonal pattern real or noise?
        
        Returns:
        - p_value: Significance of seasonal effect
        - effect_size: Magnitude of seasonal deviation
        - best_months: Months with highest returns
        - worst_months: Months with lowest returns
        """
    
    def plot_seasonal_heatmap(self, ticker: str):
        """Heatmap of returns by month × year"""

# Usage
analyzer = SeasonalityAnalyzer()
ng_pattern = analyzer.calculate_seasonal_pattern('NG')

# Show that NG prices tend to rise in summer (low), peak in winter (high)
print(f"Average summer return (Jun-Aug): {ng_pattern[152:244].mean():.2%}")
print(f"Average winter return (Dec-Feb): {ng_pattern[[335, 365, 1, 31, 32, 59]].mean():.2%}")
```

**Integration with Signal Generation:**
```python
class AdaptiveSeasonalSignal(SignalModel):
    """
    Use seasonality as signal modifier, not standalone signal.
    
    Logic:
    - Base signal (momentum, mean reversion)
    - If in favorable seasonal period: Increase position size
    - If in unfavorable period: Reduce position size or skip
    """
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Get base signal
        base_signal = self.base_strategy.generate(df)
        
        # Get seasonal adjustment
        seasonal_pattern = self.seasonality_analyzer.get_pattern(self.ticker)
        df['seasonal_adj'] = df['Date'].dt.dayofyear.map(seasonal_pattern)
        
        # Adjust signal strength
        df['Signal'] = base_signal['Signal'] * (1 + df['seasonal_adj'])
        
        return df
```

---

## 🎯 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2) - **START HERE**

**Objective:** Build core infrastructure for multi-asset support

**Week 1: Asset Registry & Metadata**
- [ ] Create `core/asset_registry.py`
  - Define `AssetType`, `AssetClass`, `AssetMetadata` dataclasses
  - Build `ASSET_REGISTRY` with 15-20 assets:
    - Equity futures: ES, NQ, RTY, YM
    - Energy: CL, NG, RB (gasoline), HO (heating oil)
    - Metals: GC, SI (silver), HG, PL (platinum)
    - Agriculture: ZC (corn), ZS (soybeans), ZW (wheat)
    - Fixed income: ZN (10Y), ZB (30Y)
  - Add helper functions: `get_asset()`, `filter_by_class()`, `requires_rollover()`
  
- [ ] Update `MultiAssetLoader` to use registry
  - Replace hard-coded `ASSET_FILES` with `ASSET_REGISTRY`
  - Auto-discover yfinance symbols from registry
  - Validate tickers against registry on load

**Week 2: Futures Rollover Foundation**
- [ ] Create `core/futures/rollover_handler.py`
  - Build expiration calendar for CME, NYMEX, ICE
  - Implement `get_active_contract()` logic
  - Create `create_continuous_series()` with Panama method
  - Test on historical ES data (known rollover dates)

- [ ] Create `core/futures/rollover_config.py`
  - Define rollover rules per asset (days before expiry, volume-based, etc.)
  - Store rollover cost tracking

- [ ] Test rollover on sample data
  - Download 2 years of CL front month + 2nd month
  - Generate continuous series
  - Verify no gaps on rollover dates
  - Compare our continuous series vs. CL=F from yfinance

**Deliverables:**
- ✅ `ASSET_REGISTRY` with 15+ assets
- ✅ Rollover handler generating continuous series
- ✅ MultiAssetLoader integrated with registry
- ✅ Unit tests for rollover logic

---

### Phase 2: Data Infrastructure (Weeks 3-4)

**Objective:** Load commodity data with rollover support

**Week 3: Commodity Data Loading**
- [ ] Extend `MultiAssetLoader.load_assets()`:
  - Add `use_continuous=True` kwarg
  - Add `rollover_method='panama'` kwarg
  - For futures, call `rollover_handler.create_continuous_series()`
  - Return both continuous AND individual contracts (for spread strategies)

- [ ] Download commodity data:
  - CL, NG, RB, HO (energy)
  - GC, SI, HG (metals)
  - ZC, ZS, ZW (agriculture)
  - Store in `Dataset/energy/`, `Dataset/metals/`, `Dataset/agriculture/`

- [ ] Test multi-commodity loading:
  ```python
  data = load_assets(['ES', 'CL', 'GC', 'NG'], use_continuous=True)
  # Verify all aligned to same dates
  # Verify no NaN on rollover dates
  ```

**Week 4: Term Structure Support**
- [ ] Add `load_term_structure()` to loader:
  ```python
  term_structure = loader.load_term_structure('CL', date='2024-12-01', num_contracts=3)
  # Returns: front_month, second_month, third_month prices
  ```

- [ ] Create helper for spread strategies:
  ```python
  spread_data = loader.create_spread_series('CL', method='calendar')
  # Returns: front_month, second_month, spread (front - second)
  ```

**Deliverables:**
- ✅ 10+ commodities loaded with rollover
- ✅ Term structure data available
- ✅ Spread data structure defined
- ✅ Validated against known historical rollover dates

---

### Phase 3: Commodity Signals (Weeks 5-6)

**Objective:** Build commodity-specific signal generators

**Week 5: Carry & Seasonal Signals**
- [ ] Create `signals/commodities/carry.py`:
  - `CarrySignal`: Long backwardation, short contango
  - `RollYieldSignal`: Exploit predictable roll costs

- [ ] Create `signals/commodities/seasonal.py`:
  - `SeasonalityAnalyzer`: Calculate historical patterns
  - `SeasonalSignal`: Trade based on month/day-of-year
  - `AdaptiveSeasonalSignal`: Combine with base strategy

- [ ] Test on natural gas (NG):
  - Known winter seasonality (high prices Dec-Feb)
  - Validate: Long positions in summer outperform winter longs?

**Week 6: Spread Signals**
- [ ] Create `signals/commodities/spreads.py`:
  - `CalendarSpreadSignal`: Front vs. back month mean reversion
  - `InterCommoditySpreadSignal`: Related assets (CL vs. RB)
  - `CrackSpreadSignal`: Refining margin arbitrage

- [ ] Create example spread strategies:
  - NG calendar spread (winter vs. summer)
  - Crack spread (3:2:1 crude:gasoline:heating oil)

**Deliverables:**
- ✅ 5+ commodity-specific signal generators
- ✅ Backtest results on NG seasonal strategy
- ✅ Spread strategy examples working
- ✅ Documentation of each signal logic

---

### Phase 4: Intraday Support (Weeks 7-8)

**Objective:** Enable high-frequency commodity strategies

**Week 7: Intraday Data Infrastructure**
- [ ] Create `core/intraday_loader.py`:
  - Load hourly bars from Polygon.io or Alpha Vantage
  - Load minute bars (if needed)
  - Resample to daily for comparison

- [ ] Download electricity data:
  - EIA API for day-ahead prices (free!)
  - PJM, ERCOT, CAISO markets
  - Store hourly data for 2020-2025

- [ ] Test intraday loading:
  ```python
  ng_hourly = load_intraday('NG', start='2024-01-01', interval='1h')
  # Should have 24 rows per day
  ```

**Week 8: Intraday Signals**
- [ ] Create `signals/energy_specific/electricity_intraday.py`:
  - `TimeOfDayMeanReversion`: Within-day reversion
  - `PeakOffPeakSpread`: Trade peak vs. off-peak hours
  - `HourlyMomentum`: Momentum at hourly frequency

- [ ] Test electricity strategy:
  - Use real EIA hourly data
  - Validate: Does price spike at 5-6pm (peak demand)?
  - Does mean reversion work within day?

**Deliverables:**
- ✅ Intraday data loader
- ✅ Electricity hourly data (3+ years)
- ✅ 2+ intraday signal strategies
- ✅ Backtest showing intraday edge (if exists)

---

### Phase 5: Portfolio Integration (Weeks 9-10)

**Objective:** Integrate commodities into multi-asset portfolio

**Week 9: Position Sizing for Futures**
- [ ] Update `core/portfolio/position_sizers.py`:
  - Add `ContractMultiplierSizer`
  - Add `MarginAwareSizer`
  - Integrate with `ASSET_REGISTRY` for contract specs

- [ ] Update `PortfolioManagerV2`:
  - Check `asset_type` when sizing positions
  - Use contract-aware sizer for futures
  - Track margin usage separately from capital

- [ ] Test position sizing:
  ```python
  # CL at $70, want $10,000 exposure
  # Should return: 0 or 1 contracts (not fractional)
  ```

**Week 10: Multi-Asset Portfolio**
- [ ] Create sample multi-asset portfolio:
  - 30% equity futures (ES, NQ)
  - 30% energy (CL, NG)
  - 20% metals (GC, HG)
  - 20% fixed income (ZN, ZB)

- [ ] Run walk-forward validation:
  - Each strategy on its asset class
  - Compare diversified portfolio vs. single-asset

- [ ] Generate portfolio report:
  - Show strategy attribution by asset class
  - Correlation matrix (equity vs. commodity vs. fixed income)
  - Sharpe improvement from diversification

**Deliverables:**
- ✅ Contract-aware position sizing working
- ✅ Multi-asset portfolio (4+ asset classes)
- ✅ Walk-forward results with commodities
- ✅ Portfolio report with diversification analysis

---

### Phase 6: Polish & Validation (Weeks 11-12)

**Objective:** Production-ready system with full documentation

**Week 11: Testing & Validation**
- [ ] Unit tests for all components:
  - Asset registry validation
  - Rollover handler edge cases
  - Commodity signal generators
  - Position sizing with futures

- [ ] Integration tests:
  - Full backtest ES + CL + GC (3 asset classes)
  - Verify rollover doesn't create gaps
  - Verify position sizing respects margins

- [ ] Performance validation:
  - Run 10-year backtest on all assets
  - Statistical significance tests
  - Out-of-sample validation (2024-2025)

**Week 12: Documentation & Examples**
- [ ] Update README with commodity examples
- [ ] Create tutorial notebook:
  - "How to add a new commodity"
  - "Building a seasonal natural gas strategy"
  - "Trading calendar spreads"

- [ ] Architecture documentation:
  - System flow diagram
  - Asset registry design
  - Rollover handler internals

- [ ] API documentation:
  - All new classes and methods
  - Usage examples for each signal

**Deliverables:**
- ✅ 90%+ test coverage on new code
- ✅ Full documentation
- ✅ Tutorial notebooks
- ✅ Production-ready system

---

## 🧩 Key Design Decisions

### 1. **Continuous vs. Individual Contracts**

**Decision:** Support BOTH

**Rationale:**
- Most strategies need continuous series (trend following, momentum)
- Spread strategies need individual contracts (front vs. back)
- Rollover analysis needs both (validate our adjustments)

**Implementation:**
```python
# Default: continuous series
data = load_assets(['CL'], use_continuous=True)

# For spreads: get term structure
term_structure = loader.load_term_structure('CL', date, num_contracts=3)

# For validation: get individual contracts
individual = loader.load_individual_contracts('CL', contracts=['CLZ24', 'CLF25'])
```

---

### 2. **Rollover Timing**

**Decision:** Configurable per asset, default to 5-7 days before expiration

**Rationale:**
- Different markets have different liquidity patterns
- Energy (CL, NG): Roll 5-7 days before (thick liquidity in front 2 months)
- Agriculture: May need more days (less liquid)
- Equity futures: Can roll closer to expiry (very liquid)

**Implementation:**
```python
ROLLOVER_RULES = {
    'CL': {'method': 'days_before_expiry', 'days': 5},
    'NG': {'method': 'days_before_expiry', 'days': 7},
    'ES': {'method': 'volume_based', 'threshold': 1.0},  # Roll when back > front volume
}
```

---

### 3. **Margin vs. Capital Tracking**

**Decision:** Track separately

**Rationale:**
- Futures use margin (5-10% of notional)
- Stocks/ETFs use full capital
- Need to ensure we don't over-leverage futures

**Implementation:**
```python
class PortfolioManagerV2:
    def __init__(self):
        self.capital = 1_000_000  # Total capital
        self.margin_used = 0      # Margin for open futures positions
        self.margin_available = 0.5 * self.capital  # Max 50% capital at risk as margin
```

---

### 4. **Intraday vs. Daily Strategies**

**Decision:** Separate but compatible

**Rationale:**
- Different data frequencies (daily vs. hourly/minute)
- Different execution (EOD close vs. intraday limit orders)
- But should share same portfolio manager, risk controls

**Implementation:**
```python
# Daily strategy
daily_signal = MomentumSignal()
daily_result = pm.run_backtest(signals={'CL_daily': daily_signal}, prices={'CL_daily': daily_prices})

# Intraday strategy
intraday_signal = TimeOfDayMeanReversion()
intraday_result = pm.run_backtest(signals={'NG_intraday': intraday_signal}, prices={'NG_intraday': hourly_prices})

# Combined portfolio
combined_result = pm.combine_strategies([daily_result, intraday_result])
```

---

### 5. **Asset Registry Extensibility**

**Decision:** Registry is a Python dict (not database) but with validation

**Rationale:**
- Easy to add new assets (just add to dict)
- No database setup required (solo developer)
- Validated via dataclasses (type checking)
- Can export to JSON/CSV if needed later

**Implementation:**
```python
# Adding new asset is just:
ASSET_REGISTRY['BTC'] = AssetMetadata(
    ticker='BTC',
    name='Bitcoin Futures',
    asset_type=AssetType.FUTURES,
    asset_class=AssetClass.CURRENCY,
    contract_multiplier=5,  # 5 BTC per contract
    ...
)

# Validation happens automatically (dataclass + type hints)
```

---

## 🎓 Important Considerations

### 1. **Data Quality for Commodities**

**Challenge:** Commodity futures data is harder to get than equity data

**Solutions:**
- **Free sources:**
  - yfinance: CL=F, GC=F, NG=F (continuous, but may have gaps)
  - EIA API: Electricity, natural gas, oil (free government data)
  - Quandl: Some commodity data (free tier limited)
  
- **Paid sources (if needed later):**
  - Interactive Brokers: Historical data ($$$)
  - CSI Data: Comprehensive futures database ($$$)
  - Polygon.io: Commodities + intraday ($$)

**Recommendation for now:**
- Start with yfinance for liquid futures (CL, GC, NG)
- Use EIA API for electricity/gas (it's free!)
- Add paid sources later if needed

---

### 2. **Rollover Cost Reality**

**Challenge:** Rollover is NOT free - you pay the spread between front and back month

**Example:**
```
Front month (Dec CL): $70.50
Back month (Jan CL):  $71.00
Spread: $0.50/barrel
Cost per contract: $0.50 × 1,000 barrels = $500

If rolling 10 contracts: $5,000 cost!
```

**Why it matters:**
- Contango (back > front): You PAY to roll (negative roll yield)
- Backwardation (front > back): You EARN on roll (positive roll yield)
- This can be 5-10% annualized for some commodities

**Our implementation:**
```python
def calculate_rollover_cost(self, front_price, back_price, num_contracts, multiplier):
    spread = back_price - front_price
    cost_per_contract = spread * multiplier
    total_cost = cost_per_contract * num_contracts
    return total_cost

# Track in backtest
portfolio.apply_rollover_cost(total_cost)
```

---

### 3. **Seasonality Overfitting Risk**

**Challenge:** Seasonal patterns can break (climate change, tech shifts, etc.)

**Mitigation:**
- Don't use seasonality as standalone signal (too weak)
- Use as signal MODIFIER (boost momentum in favorable season)
- Require statistical significance (p < 0.05)
- Out-of-sample validation (test pattern on recent years)

**Example:**
```python
# BAD: Pure seasonal signal
signal = 1 if month in [6, 7, 8] else 0  # Always long in summer

# GOOD: Seasonal modifier
base_signal = momentum_signal.generate(df)
seasonal_boost = 1.2 if month in favorable_months else 1.0
final_signal = base_signal * seasonal_boost
```

---

### 4. **Electricity Trading Complexity**

**Challenge:** Electricity markets are VERY different from other commodities

**Key differences:**
- **No storage:** Can't "hold" electricity
- **Regional:** Different prices in different grids (PJM vs. ERCOT vs. CAISO)
- **Volatile:** Prices can spike 10-100x during peak demand
- **Predictable:** Strong time-of-day and seasonal patterns
- **Weather-driven:** Heat waves / cold snaps drive demand

**Implications:**
- Can't do buy-and-hold strategies (no storage)
- Must trade day-ahead or real-time markets
- Need weather forecasts for signals
- Risk management critical (price spikes can be devastating)

**Our approach:**
- Start with day-ahead market (1-day forward)
- Focus on mean reversion within day (spikes revert quickly)
- Use historical patterns (peak vs. off-peak)
- Add weather signals later (optional)

---

### 5. **Carry Trade Mechanics**

**Challenge:** Understanding when carry strategies work

**Carry in commodities:**
```
Carry = (Back Month - Front Month) / Front Month

Positive carry (backwardation): Front > Back
- Means market expects prices to FALL
- You earn "roll yield" by holding front month
- Common in oil during supply disruptions

Negative carry (contango): Back > Front
- Means market expects prices to RISE (or storage costs high)
- You PAY to roll (negative roll yield)
- Common in gold, grains (storage costs)
```

**When to use:**
- Backwardation: LONG the commodity (earn carry)
- Contango: SHORT or avoid (pay carry)
- Calendar spread: If contango too steep, bet on convergence

---

### 6. **Inter-Commodity Spreads**

**Challenge:** Which commodity pairs make sense?

**Logical spreads:**

1. **Crack Spread** (Oil Refining)
   ```
   3:2:1 Crack = (2 × Gasoline + 1 × Heating Oil) / 3 - Crude Oil
   
   Wide spread → Refiners profitable → Buy crude to refine
   Narrow spread → Refiners losing money → Sell crude
   ```

2. **Spark Spread** (Natural Gas Power Generation)
   ```
   Spark Spread = Electricity Price - (Natural Gas Price × Heat Rate)
   
   Wide spread → Power plants profitable → Buy gas to generate
   Narrow spread → Power plants losing money → Sell gas
   ```

3. **Crush Spread** (Soybean Processing)
   ```
   Crush = (Soybean Meal + Soybean Oil) - Soybeans
   
   Wide crush → Processors profitable → Buy soybeans to crush
   Narrow crush → Processors losing money
   ```

**Implementation:**
```python
class CrackSpreadSignal(SignalModel):
    def generate(self, prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        cl = prices['CL']['Close']
        rb = prices['RB']['Close']  # Gasoline
        ho = prices['HO']['Close']  # Heating oil
        
        crack = (2 * rb + ho) / 3 - cl
        crack_zscore = (crack - crack.rolling(60).mean()) / crack.rolling(60).std()
        
        signal = np.where(crack_zscore < -2, 1,   # Crack too narrow, long crude
                         np.where(crack_zscore > 2, -1,  # Crack too wide, short crude
                                 0))
        return signal
```

---

## 📊 Success Metrics

### Technical Goals
- [ ] **Asset coverage:** 20+ assets across 5 asset classes
- [ ] **Rollover accuracy:** <0.1% price gap on rollover dates
- [ ] **Data completeness:** 95%+ valid data after rollover adjustments
- [ ] **Position sizing:** Contract-aware, respects margin limits
- [ ] **Test coverage:** 85%+ on new commodity code

### Performance Goals (Portfolio Level)
- [ ] **Sharpe Ratio:** > 1.0 (commodities + equities combined)
- [ ] **Diversification benefit:** Portfolio Sharpe > max(individual strategies)
- [ ] **Drawdown:** < 25% max drawdown
- [ ] **Robustness:** Positive returns in 70%+ of walk-forward folds
- [ ] **Statistical significance:** p < 0.05 on permutation tests

### Strategy-Specific Goals
- [ ] **NG seasonal:** Exploit winter/summer pattern (Sharpe > 0.5)
- [ ] **CL carry:** Profit from backwardation periods (win rate > 55%)
- [ ] **Electricity intraday:** Mean reversion edge (if exists)
- [ ] **Crack spread:** Refining margin trades profitable (Sharpe > 0.3)

---

## 🚀 Next Steps: 2-Week Sprint Plan

### Week 1: Asset Registry + Rollover Foundation

**Day 1-2: Asset Registry**
```python
# Create core/asset_registry.py
# Define enums and AssetMetadata dataclass
# Build ASSET_REGISTRY with 15+ assets
# Add validation and query helpers
```

**Day 3-4: Rollover Handler**
```python
# Create core/futures/rollover_handler.py
# Implement expiration calendar
# Build create_continuous_series() with Panama method
# Test on ES data
```

**Day 5: Integration**
```python
# Update MultiAssetLoader to use registry
# Test loading CL with rollover
# Validate continuous series has no gaps
```

### Week 2: Commodity Signals + Data

**Day 1-2: Download Data**
```bash
# Download CL, NG, GC data (yfinance)
# Store in Dataset/energy/, Dataset/metals/
# Verify data quality
```

**Day 3-4: First Commodity Signal**
```python
# Create signals/commodities/seasonal.py
# Implement SeasonalSignal for NG
# Backtest NG seasonal strategy
# Document results
```

**Day 5: Documentation**
```markdown
# This document (COMMODITIES_EXPANSION_PLAN.md)
# Update README with commodity examples
# Create tutorial: "Adding a new commodity"
```

---

## 📝 Open Questions for Discussion

### 1. **Data Source Priority**
**Question:** Which data sources should we prioritize?
- A. Free only (yfinance + EIA) - fastest to implement
- B. Mix of free + paid (add Polygon.io for intraday) - better quality
- C. Focus on one asset class first (just energy) - depth over breadth

**Recommendation:** Start with A (free sources), add B if strategies show promise.

---

### 2. **Asset Class Priority**
**Question:** Which commodities to tackle first?
- A. Energy (CL, NG) - liquid, strong patterns, good for learning
- B. Metals (GC, HG) - also liquid, less complex than energy
- C. Agriculture (ZC, ZS) - seasonal patterns, but less liquid

**Recommendation:** Energy first (CL, NG) - most liquid, best data availability, strongest patterns.

---

### 3. **Intraday vs. Daily**
**Question:** Should we build intraday support now or later?
- A. Now - enables electricity strategies (unique edge)
- B. Later - daily strategies first (simpler, proven patterns)

**Recommendation:** Later. Master daily commodities first (rollover, carry, seasonality), then add intraday as Phase 4.

---

### 4. **Rollover Method**
**Question:** Which rollover adjustment method to use as default?
- A. Panama (adjust historical prices by roll difference)
- B. Ratio (adjust by roll ratio)
- C. Both (let user choose)

**Recommendation:** C (both available), default to Panama (most common in industry).

---

### 5. **Testing Strategy**
**Question:** How rigorous should testing be during development?
- A. Full TDD (test-driven development) - slower, robust
- B. Prototype first, test after - faster, riskier
- C. Hybrid (tests for core logic, prototype for signals)

**Recommendation:** C (hybrid). Test rollover handler thoroughly (critical), prototype signals quickly (iterative).

---

## 🎯 Final Thoughts

### The Big Picture
This expansion transforms your system from a **single-asset equity futures trader** into a **true multi-asset quantitative platform**. The key insight is that commodities are NOT just "another asset" - they have unique characteristics (rollover, carry, seasonality) that require specialized infrastructure.

### Core Philosophy
**"One system, any asset class, same rigor"**
- Equity futures: Trend following, momentum
- Energy: Seasonality, carry, spreads
- Metals: Safe haven, macro driven
- Agriculture: Weather, crop cycles
- Fixed income: Rates, curve strategies

All using the same portfolio manager, risk controls, backtesting framework, and validation methodology.

### What Makes This Plan Good?

1. **Incremental:** Build foundation first (registry, rollover), then signals
2. **Pragmatic:** Start with free data, add paid sources if needed
3. **Testable:** Each component independently verifiable
4. **Extensible:** Add new assets by updating registry (no code changes)
5. **Realistic:** Models rollover costs, margin requirements, spreads
6. **Flexible:** Supports both daily and intraday strategies
7. **Professional:** Same rigor as institutional quant shop

### Success Criteria
You'll know this is working when you can:
- Add a new commodity (e.g., soybeans) in 5 minutes (just update registry)
- Backtest a seasonal natural gas strategy in one notebook
- Run a diversified portfolio (equities + energy + metals) with proper rollover
- See Sharpe improvement from cross-asset diversification
- Trust the results (validated rollover, realistic costs)

---

**Ready to discuss and refine before implementation? 🚀**

---

## Appendix: Example Usage (After Implementation)

### Example 1: Simple Commodity Strategy
```python
from core.asset_registry import ASSET_REGISTRY
from core.multi_asset_loader import load_assets
from signals.commodities.seasonal import SeasonalSignal
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2

# Load natural gas data (automatically handles rollover)
data = load_assets(['NG'], start_date='2015-01-01', use_continuous=True)

# Create seasonal signal
signal = SeasonalSignal(ticker='NG', favorable_months=[6, 7, 8, 9])

# Generate signals
ng_signals = signal.generate(data['NG'].reset_index())

# Run backtest
pm = PortfolioManagerV2(initial_capital=100_000)
result = pm.run_backtest(
    signals={'NG': ng_signals[['Signal']]},
    prices={'NG': data['NG']}
)

print(f"Sharpe: {result.metrics['Sharpe Ratio']:.2f}")
print(f"CAGR: {result.metrics['CAGR']:.2%}")
```

### Example 2: Multi-Asset Portfolio
```python
# Load multiple asset classes
data = load_assets(
    ['ES', 'NQ', 'CL', 'NG', 'GC', 'HG'],
    start_date='2010-01-01',
    use_continuous=True,
    rollover_method='panama'
)

# Different strategies for different assets
strategies = {
    'ES': MomentumSignal(lookback=60),
    'NQ': TrendFollowingLongShort(fast=20, slow=100),
    'CL': CarrySignal(threshold=0.02),
    'NG': SeasonalSignal(favorable_months=[6, 7, 8]),
    'GC': MeanReversionSignal(lookback=20),
    'HG': MomentumSignal(lookback=40)
}

# Generate all signals
signals = {}
for ticker, strategy in strategies.items():
    signals[ticker] = strategy.generate(data[ticker].reset_index())

# Run portfolio backtest
result = pm.run_backtest(signals=signals, prices=data)

# Portfolio attribution
print("\nStrategy Attribution:")
for ticker in ['ES', 'NQ', 'CL', 'NG', 'GC', 'HG']:
    contrib = result.get_strategy_contribution(ticker)
    print(f"{ticker}: {contrib:.2%} of total return")
```

### Example 3: Calendar Spread Strategy
```python
from signals.commodities.spreads import CalendarSpreadSignal

# Load term structure data
cl_term = loader.load_term_structure('CL', num_contracts=2)

# Create spread signal
spread_signal = CalendarSpreadSignal(
    spread_threshold=2.0,  # Enter when z-score > 2
    lookback=60
)

# Generate signals (trade the spread)
spread_signals = spread_signal.generate(cl_term)

# Backtest (long front, short back when spread too wide)
result = pm.run_backtest(
    signals={'CL_spread': spread_signals},
    prices={'CL_front': cl_term['front_month'], 'CL_back': cl_term['second_month']}
)
```

---

**Document Version:** 1.0  
**Last Updated:** December 1, 2025  
**Status:** Ready for discussion and refinement
