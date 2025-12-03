# Oil & Gas Trading Implementation Plan

**Date:** December 1, 2025  
**Focus:** CL (Crude Oil) + NG (Natural Gas) with daily data  
**Philosophy:** Reusable infrastructure, notebook for exploration, vectorization later

---

## 🎯 Objectives

1. **Trade CL and NG futures** with proper rollover handling
2. **Reuse existing code** wherever possible (signals, portfolio manager, backtester)
3. **Build modular components** that work for ALL futures (equity + commodity)
4. **Work through issues in notebook**, refactor stable code to modules
5. **Get it working first**, optimize later (when we add intraday data)

---

## 🏗️ Architecture: What to Build

### 1. ✅ Keep Using (Already Works)
- `PortfolioManagerV2` - Works for any asset
- `SignalModel` base class - Works for any asset
- `MomentumSignal`, `TrendFollowingLongShort`, `MeanReversionSignal` - Work for commodities too!
- `Reporter`, `RiskDashboard` - Asset-agnostic
- Walk-forward validation framework - Asset-agnostic
- Position sizers (`FixedFractionalSizer`, `ATRSizer`, etc.) - Just need minor enhancement

### 2. 🔧 Extend (Minor Changes)
- `MultiAssetLoader` - Add rollover integration, keep same API
- Position sizers - Add contract multiplier awareness (one method)
- `BacktestResult` - Add rollover cost tracking (optional field)

### 3. 🆕 Build New (Required)
- `AssetRegistry` - Metadata for all assets (simple dict + dataclass)
- `FuturesRolloverHandler` - Contract expiration + continuous series generation
- `core/futures/` module - Houses rollover logic

---

## 📦 Component Details

### Component 1: Asset Registry

**File:** `core/asset_registry.py`

**Purpose:** Single source of truth for asset specifications

**What it contains:**
```python
# Enums
AssetType(Enum): FUTURES, STOCK, ETF, INDEX, FOREX
AssetClass(Enum): EQUITY, COMMODITY_ENERGY, COMMODITY_METAL, etc.

# Data structure
@dataclass
class AssetMetadata:
    ticker: str
    name: str
    asset_type: AssetType
    asset_class: AssetClass
    contract_multiplier: Optional[float] = None  # For futures
    tick_size: Optional[float] = None
    expiration_cycle: Optional[str] = None  # 'monthly', 'quarterly'
    requires_rollover: bool = False
    yfinance_symbol: Optional[str] = None
    typical_margin_pct: Optional[float] = None

# Registry
ASSET_REGISTRY: Dict[str, AssetMetadata] = {
    'ES': AssetMetadata(ticker='ES', name='S&P 500 E-mini', asset_type=AssetType.FUTURES, ...),
    'CL': AssetMetadata(ticker='CL', name='WTI Crude Oil', asset_type=AssetType.FUTURES, ...),
    'NG': AssetMetadata(ticker='NG', name='Natural Gas', asset_type=AssetType.FUTURES, ...),
    # ... more assets
}

# Helper functions
def get_asset(ticker: str) -> AssetMetadata
def filter_by_class(asset_class: AssetClass) -> List[AssetMetadata]
def requires_rollover(ticker: str) -> bool
```

**Why it's reusable:**
- Works for equity futures (ES, NQ, GC) AND commodities (CL, NG)
- Easy to add new assets (just add to dict)
- No code changes needed to support new asset classes

**Estimated effort:** 2-3 hours

---

### Component 2: Futures Rollover Handler

**File:** `core/futures/rollover_handler.py`

**Purpose:** Handle contract expiration and create continuous price series

**Key features:**
```python
class FuturesRolloverHandler:
    """
    Handles futures contract rollovers for ALL futures (equity + commodity).
    """
    
    def __init__(self, asset_registry: Dict[str, AssetMetadata]):
        self.registry = asset_registry
        self.expiration_calendars = self._load_expiration_calendars()
    
    def get_active_contract(self, ticker: str, date: pd.Timestamp) -> str:
        """
        Return active contract for given date.
        
        Example:
        - get_active_contract('CL', '2024-11-01') -> 'CLZ24' (Dec 2024)
        - get_active_contract('CL', '2024-11-20') -> 'CLF25' (Jan 2025, rolled early)
        """
    
    def get_rollover_dates(self, ticker: str, start: str, end: str) -> List[pd.Timestamp]:
        """
        Get all rollover dates in period.
        
        Example:
        - get_rollover_dates('CL', '2020-01-01', '2024-12-31') 
          -> [2020-01-15, 2020-02-14, ..., 2024-11-15, 2024-12-13]
        """
    
    def create_continuous_series(
        self, 
        ticker: str, 
        contract_data: Dict[str, pd.DataFrame],  # {'CLZ24': df, 'CLF25': df, ...}
        method: str = 'panama'
    ) -> pd.DataFrame:
        """
        Build continuous price series from individual contracts.
        
        Methods:
        - 'panama': Adjust historical prices by roll difference (most common)
        - 'ratio': Adjust by roll ratio (for percentage strategies)
        - 'none': No adjustment (for spread trading)
        
        Returns: DataFrame with continuous OHLC prices (no gaps on rollover dates)
        """
    
    def calculate_rollover_cost(
        self, 
        front_price: float, 
        back_price: float, 
        contracts: int,
        multiplier: float
    ) -> float:
        """
        Calculate cost of rolling from front to back month.
        
        Example:
        - Front (Dec CL): $70.50
        - Back (Jan CL): $71.00
        - Spread: $0.50/barrel
        - Contracts: 10
        - Multiplier: 1000 barrels/contract
        - Cost: $0.50 × 1000 × 10 = $5,000 (negative carry)
        """
```

**Expiration Calendar (initially simplified):**
```python
# Store as dict of rules per ticker
EXPIRATION_RULES = {
    'CL': {
        'cycle': 'monthly',  # Every month
        'day_rule': 'third_business_day_before_25th',
        'roll_days_before': 5  # Roll 5 days before expiration
    },
    'NG': {
        'cycle': 'monthly',
        'day_rule': 'third_business_day_before_end',
        'roll_days_before': 7
    },
    'ES': {
        'cycle': 'quarterly',  # Mar, Jun, Sep, Dec
        'day_rule': 'third_friday',
        'roll_days_before': 5
    },
    'GC': {
        'cycle': 'monthly',
        'day_rule': 'third_last_business_day',
        'roll_days_before': 5
    }
}
```

**Why it's reusable:**
- Works for ALL futures (ES, NQ, GC, CL, NG, etc.)
- Pluggable rules per asset
- Same interface regardless of commodity type

**Estimated effort:** 6-8 hours (most complex component)

---

### Component 3: Enhanced MultiAssetLoader

**File:** `core/multi_asset_loader.py` (extend existing)

**Changes needed:**
```python
class MultiAssetLoader:
    """Extended to support futures rollover"""
    
    def __init__(
        self, 
        dataset_dir: Optional[Path] = None, 
        use_yfinance: bool = True,
        rollover_handler: Optional[FuturesRolloverHandler] = None,  # NEW
        asset_registry: Optional[Dict[str, AssetMetadata]] = None  # NEW
    ):
        self.dataset_dir = dataset_dir or DATASET_DIR
        self.use_yfinance = use_yfinance
        self.rollover_handler = rollover_handler  # NEW
        self.registry = asset_registry  # NEW
    
    def load_assets(
        self, 
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_continuous: bool = True,  # NEW - use continuous series for futures
        rollover_method: str = 'panama',  # NEW
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Load assets with automatic rollover handling for futures.
        
        For non-futures (stocks, ETFs): Works as before
        For futures: Applies rollover adjustment if use_continuous=True
        """
        
        # Existing loading logic...
        
        # NEW: Apply rollover if needed
        for ticker, df in raw_data.items():
            if self.registry and self.rollover_handler:
                metadata = self.registry.get(ticker)
                if metadata and metadata.requires_rollover and use_continuous:
                    # Create continuous series
                    df = self.rollover_handler.create_continuous_series(
                        ticker, 
                        {ticker: df},  # For now, assume single contract
                        method=rollover_method
                    )
            aligned_data[ticker] = df
        
        return aligned_data
```

**Why minimal changes:**
- Existing API stays the same (backwards compatible)
- Rollover is opt-in (`use_continuous=True`)
- Falls back to old behavior if no rollover_handler provided

**Estimated effort:** 2-3 hours

---

### Component 4: Contract-Aware Position Sizing

**File:** `core/portfolio/position_sizers.py` (extend existing)

**Changes needed:**
```python
class ContractAwareSizer:
    """
    Mixin for position sizers to respect futures contract specifications.
    """
    
    def adjust_for_contract_size(
        self, 
        target_dollar_amount: float,
        price: float,
        metadata: AssetMetadata
    ) -> float:
        """
        Convert dollar amount to number of contracts.
        
        Returns: Number of contracts (must be integer)
        """
        if metadata.contract_multiplier is None:
            # Stock/ETF: return dollar amount / price
            return target_dollar_amount / price
        else:
            # Futures: calculate contracts
            notional_per_contract = price * metadata.contract_multiplier
            contracts = target_dollar_amount / notional_per_contract
            return round(contracts)  # Must be whole number

# Extend existing sizers
class FixedFractionalSizer:
    """Extend with contract awareness"""
    
    def calculate_position_size(
        self, 
        capital: float, 
        price: float, 
        metadata: Optional[AssetMetadata] = None  # NEW parameter
    ) -> float:
        """Calculate position size (shares or contracts)"""
        
        # Existing logic
        target_dollar = capital * self.max_position_pct
        
        # NEW: Adjust for contract size if futures
        if metadata and metadata.contract_multiplier:
            return ContractAwareSizer.adjust_for_contract_size(
                target_dollar, price, metadata
            )
        else:
            return target_dollar / price
```

**Why minimal changes:**
- Just add one optional parameter (`metadata`)
- Backwards compatible (if no metadata, works as before)
- All existing sizers get same enhancement

**Estimated effort:** 1-2 hours

---

## 📓 Notebook-Driven Development

### Notebook: `oil_gas_exploration.ipynb`

**Purpose:** Test and validate all components before finalizing

**Structure:**
```markdown
# Oil & Gas Trading System Development

## 1. Setup & Imports
- Import all modules
- Load asset registry
- Initialize rollover handler

## 2. Data Loading Tests
- Load CL data (test rollover)
- Load NG data (test rollover)
- Validate continuous series (no gaps)
- Compare our continuous vs. yfinance continuous

## 3. Rollover Validation
- Check rollover dates match CME calendar
- Verify rollover costs calculated correctly
- Test Panama vs. ratio adjustment methods

## 4. Position Sizing Tests
- Test contract-aware sizing
- Verify position sizes are integers
- Check margin calculations

## 5. Signal Generation
- Apply existing momentum signal to CL
- Apply trend following to NG
- Verify signals make sense

## 6. Backtest (Full Pipeline)
- Run backtest on CL (2015-2024)
- Run backtest on NG (2015-2024)
- Compare vs. equity futures (ES)

## 7. Multi-Asset Portfolio
- Combine ES + CL + NG
- Test portfolio manager with mixed assets
- Verify rollover costs tracked properly

## 8. Validation & Metrics
- Compare our results vs. buy-and-hold
- Check Sharpe, CAGR, max drawdown
- Verify rollover didn't introduce bugs

## 9. Issues & Refinements
- Document any problems
- Test edge cases
- Identify code to refactor to modules
```

**Workflow:**
1. **Prototype in notebook** - Quick iteration, test ideas
2. **Validate results** - Make sure it works correctly
3. **Refactor to modules** - Move stable code out of notebook
4. **Repeat** - Iterate until solid

---

## 🎯 Implementation Steps (2-3 Days)

### Day 1: Foundation
**Morning (3-4 hours):**
- [ ] Create `core/asset_registry.py`
  - Define enums and dataclass
  - Build ASSET_REGISTRY (ES, NQ, GC, CL, NG to start)
  - Add helper functions
  - Write 5-10 unit tests

**Afternoon (3-4 hours):**
- [ ] Create `core/futures/` module
  - Create `__init__.py`
  - Create `rollover_handler.py` skeleton
  - Implement expiration calendar (simple version)
  - Implement `get_rollover_dates()` for CL
  - Test: Does it match CME calendar?

### Day 2: Rollover Implementation
**Morning (3-4 hours):**
- [ ] Complete `FuturesRolloverHandler`
  - Implement `create_continuous_series()` (Panama method)
  - Implement `calculate_rollover_cost()`
  - Test on sample CL data (2 years)
  - Verify no gaps on rollover dates

**Afternoon (3-4 hours):**
- [ ] Integrate with `MultiAssetLoader`
  - Add rollover_handler parameter
  - Add use_continuous parameter
  - Test loading CL with rollover
  - Compare our continuous vs. CL=F from yfinance

### Day 3: Integration & Testing
**Morning (3-4 hours):**
- [ ] Enhance position sizers
  - Add contract_multiplier awareness
  - Test on CL (1000 barrels/contract)
  - Test on NG (10,000 MMBtu/contract)
  - Verify positions are integers

**Afternoon (3-4 hours):**
- [ ] Create `oil_gas_exploration.ipynb`
  - Load CL and NG data
  - Run momentum signal on CL
  - Run backtest (2015-2024)
  - Validate results
  - Document issues

**Evening (1-2 hours):**
- [ ] Refactor & cleanup
  - Move stable code from notebook to modules
  - Add docstrings
  - Write additional tests
  - Update README

---

## 📊 Success Criteria

### Technical Validation
- [ ] **Continuous series:** No NaN values on rollover dates
- [ ] **Rollover dates:** Match CME expiration calendar (±1 day tolerance)
- [ ] **Position sizing:** All positions are whole numbers (integers)
- [ ] **Rollover costs:** Calculated correctly (contango = negative cost)
- [ ] **Backtest runs:** No errors with CL and NG
- [ ] **Multi-asset:** Can combine ES + CL + NG in one portfolio

### Performance Validation (Optional, but good to check)
- [ ] **CL momentum:** Sharpe > 0.3 (should have some edge)
- [ ] **NG trend:** Sharpe > 0.2 (volatile, but tradeable)
- [ ] **Portfolio:** Sharpe > max(individual) (diversification benefit)

### Code Quality
- [ ] **Reusability:** Works for equity futures too (test on ES)
- [ ] **Minimal notebook code:** <50 lines per section
- [ ] **Docstrings:** All public methods documented
- [ ] **Tests:** 80%+ coverage on rollover handler

---

## 🔧 Testing Strategy

### Unit Tests (in `tests/`)
```python
# tests/test_asset_registry.py
def test_get_asset():
    """Test retrieving asset metadata"""
    metadata = get_asset('CL')
    assert metadata.ticker == 'CL'
    assert metadata.contract_multiplier == 1000

def test_filter_by_class():
    """Test filtering assets by class"""
    energy = filter_by_class(AssetClass.COMMODITY_ENERGY)
    assert 'CL' in [a.ticker for a in energy]
    assert 'NG' in [a.ticker for a in energy]

# tests/test_rollover_handler.py
def test_get_rollover_dates_cl():
    """Test CL rollover dates match CME calendar"""
    handler = FuturesRolloverHandler()
    dates = handler.get_rollover_dates('CL', '2023-01-01', '2023-12-31')
    
    # CL rolls monthly, should have ~12 dates
    assert len(dates) >= 11 and len(dates) <= 13
    
    # Check one known date (Jan 2023 CL rolled ~Dec 15, 2022)
    assert pd.Timestamp('2022-12-15') in dates or pd.Timestamp('2022-12-16') in dates

def test_create_continuous_series_no_gaps():
    """Test continuous series has no gaps on rollover dates"""
    # Load sample data
    cl_data = load_sample_cl_data()
    
    # Create continuous series
    handler = FuturesRolloverHandler()
    continuous = handler.create_continuous_series('CL', cl_data, method='panama')
    
    # Check for NaN values
    assert continuous['Close'].isna().sum() == 0
    
    # Check rollover dates specifically
    rollover_dates = handler.get_rollover_dates('CL', continuous.index[0], continuous.index[-1])
    for date in rollover_dates:
        assert not continuous.loc[date, 'Close'].isna()

def test_calculate_rollover_cost():
    """Test rollover cost calculation"""
    handler = FuturesRolloverHandler()
    
    # Contango: back > front (negative carry)
    cost = handler.calculate_rollover_cost(
        front_price=70.0,
        back_price=71.0,
        contracts=10,
        multiplier=1000
    )
    assert cost == -10000  # Pay $1/barrel × 1000 barrels × 10 contracts
    
    # Backwardation: front > back (positive carry)
    cost = handler.calculate_rollover_cost(
        front_price=71.0,
        back_price=70.0,
        contracts=10,
        multiplier=1000
    )
    assert cost == 10000  # Earn $1/barrel × 1000 barrels × 10 contracts
```

### Integration Tests (in notebook)
- Load real CL data from yfinance
- Compare our continuous series vs. CL=F
- Check price differences are reasonable (<1% per year from rollover)
- Run full backtest and verify metrics

---

## 📝 Data Requirements

### CL (Crude Oil)
- **Source:** yfinance (CL=F)
- **Period:** 2010-01-01 to present
- **Frequency:** Daily
- **Contract multiplier:** 1,000 barrels
- **Expiration:** Monthly (3rd business day before 25th of prior month)

### NG (Natural Gas)
- **Source:** yfinance (NG=F)
- **Period:** 2010-01-01 to present
- **Frequency:** Daily
- **Contract multiplier:** 10,000 MMBtu
- **Expiration:** Monthly (3rd business day before end of month)

### Storage Structure
```
Dataset/
  energy/
    cl_continuous_2010_2025.csv  # Our generated continuous series
    ng_continuous_2010_2025.csv
    rollover_dates.csv  # Record of all rollover dates and costs
```

---

## 🚫 What We're NOT Doing (Yet)

To keep scope manageable:

- ❌ **Intraday data** - Daily only for now
- ❌ **Electricity** - Defer until oil/gas works
- ❌ **Term structure** - Just front month continuous for now
- ❌ **Spread strategies** - Just directional for now
- ❌ **Seasonality signals** - Use existing signals first
- ❌ **Carry signals** - After basic infrastructure works
- ❌ **Optimization** - Get it working first, optimize later
- ❌ **ML signals** - Traditional signals first

---

## 🎓 Lessons to Apply

### From Equity Futures Experience
✅ **What worked well:**
- Vectorized signal generation (no loops)
- Walk-forward validation (avoid overfitting)
- Modular signal architecture (easy to add new strategies)
- Reporter/dashboard infrastructure (reuse as-is)
- Position sizing framework (just extend it)

✅ **What to reuse:**
- Entire `PortfolioManagerV2` - works for any asset
- All existing signals - momentum, trend following, mean reversion
- Backtesting framework - asset-agnostic
- Risk controls - stop-loss, take-profit, position limits

✅ **What to improve:**
- Add rollover handling (was missing for equity futures too!)
- Better contract specification tracking (helps ES/NQ too)
- Clearer asset metadata (benefits all assets)

---

## 🎯 Expected Outcomes

After 2-3 days of implementation:

### Code Artifacts
- ✅ `core/asset_registry.py` (~200 lines)
- ✅ `core/futures/rollover_handler.py` (~400 lines)
- ✅ Enhanced `core/multi_asset_loader.py` (+50 lines)
- ✅ Enhanced `core/portfolio/position_sizers.py` (+30 lines)
- ✅ `tests/test_asset_registry.py` (~100 lines)
- ✅ `tests/test_rollover_handler.py` (~200 lines)
- ✅ `notebooks/oil_gas_exploration.ipynb` (exploration notebook)

### Capabilities
- ✅ Load CL and NG data with automatic rollover
- ✅ Generate signals using existing signal generators
- ✅ Run backtests with proper contract sizing
- ✅ Track rollover costs in backtest results
- ✅ Combine equity futures + commodities in one portfolio

### Validation
- ✅ Continuous series matches yfinance (within 1-2%)
- ✅ Rollover dates match CME calendar
- ✅ No data gaps on rollover dates
- ✅ Position sizes are whole numbers
- ✅ Backtests run without errors

---

## 🔄 Iteration Plan

**After Day 3:**
1. Run full validation in notebook
2. Document any issues or edge cases
3. Refine rollover logic if needed
4. Add more assets (HO, RB) if time permits
5. Plan next phase (seasonality signals, term structure, etc.)

**If issues found:**
- Debug in notebook first
- Add regression tests
- Update modules
- Re-run validation

**If everything works:**
- Commit to GitHub
- Update README with examples
- Plan commodity-specific signals (carry, seasonality)
- Consider adding more commodities (metals, agriculture)

---

## 💡 Pro Tips

### Development Flow
1. **Write test first** (what should happen)
2. **Implement in notebook** (quick iteration)
3. **Validate with real data** (does it work?)
4. **Refactor to module** (make it reusable)
5. **Write unit test** (prevent regression)
6. **Document** (explain why)

### Debugging Rollover
- Print rollover dates alongside price series
- Plot continuous series with rollover markers
- Compare our prices vs. yfinance prices (should be close)
- Check for sudden jumps in price (indicates bad adjustment)

### Performance Notes
- Daily data: ~5,000 days × 3 assets = 15,000 rows (fast, no optimization needed)
- Only optimize when: >100,000 rows OR >10 second backtest time
- Premature optimization is root of all evil!

---

## ✅ Ready to Start?

**First command to run:**
```bash
cd /Users/Sakarias/QuantTrading
conda activate quant-paper-trading

# Create new module directory
mkdir -p core/futures
touch core/futures/__init__.py

# Create new test file
touch tests/test_asset_registry.py
touch tests/test_rollover_handler.py

# Create notebook
jupyter notebook notebooks/oil_gas_exploration.ipynb
```

**First file to create:**
`core/asset_registry.py` - Let's build the foundation!

---

**Document Version:** 1.0  
**Status:** Ready to implement  
**Estimated Time:** 2-3 days  
**Risk:** Low (reusing proven infrastructure)
