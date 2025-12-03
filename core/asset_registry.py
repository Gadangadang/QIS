"""
Asset Registry: Single Source of Truth for Asset Metadata

Provides comprehensive asset information for ALL tradeable instruments:
- Equity futures (ES, NQ, RTY, YM)
- Commodity futures (CL, NG, GC, HG, etc.)
- Stocks and ETFs (SPY, QQQ, AAPL, etc.)

Key features:
- Type safety via dataclasses and enums
- Contract specifications (multipliers, tick sizes)
- Expiration rules for futures
- Easy querying and filtering

Usage:
    from core.asset_registry import ASSET_REGISTRY, get_asset, filter_by_class
    
    # Get asset metadata
    cl_metadata = get_asset('CL')
    print(f"Contract size: {cl_metadata.contract_multiplier} barrels")
    
    # Filter by asset class
    energy_assets = filter_by_class(AssetClass.COMMODITY_ENERGY)
    print(f"Energy commodities: {[a.ticker for a in energy_assets]}")
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List


class AssetType(Enum):
    """Asset classification by instrument type"""
    FUTURES = "futures"
    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"
    FOREX = "forex"


class AssetClass(Enum):
    """Asset class for portfolio allocation and strategy selection"""
    EQUITY = "equity"
    COMMODITY_ENERGY = "commodity_energy"
    COMMODITY_METAL = "commodity_metal"
    COMMODITY_AGRICULTURE = "commodity_agriculture"
    FIXED_INCOME = "fixed_income"
    CURRENCY = "currency"
    VOLATILITY = "volatility"


@dataclass
class AssetMetadata:
    """
    Comprehensive asset information.
    
    Handles both traditional capital markets assets AND commodity futures.
    Provides all information needed for:
    - Data loading (yfinance symbols)
    - Position sizing (contract multipliers)
    - Risk management (margin requirements)
    - Strategy selection (seasonality, expiration cycles)
    
    Attributes:
        ticker: Internal ticker symbol (e.g., 'ES', 'CL', 'SPY')
        name: Human-readable name
        asset_type: Type of instrument (futures, stock, ETF, etc.)
        asset_class: Asset class for allocation (equity, commodity, etc.)
        contract_multiplier: For futures, dollar value per point/unit (e.g., 50 for ES)
        tick_size: Minimum price increment
        expiration_cycle: 'monthly', 'quarterly', or None
        requires_rollover: Whether contract needs periodic rolling
        yfinance_symbol: Symbol for yfinance data fetching
        trading_hours: Trading session type
        typical_margin_pct: Typical margin requirement as fraction of notional
        has_seasonality: Whether asset exhibits seasonal patterns
        seasonality_pattern: Description of seasonal behavior (if applicable)
    """
    ticker: str
    name: str
    asset_type: AssetType
    asset_class: AssetClass
    
    # Futures-specific fields (None for stocks/ETFs)
    contract_multiplier: Optional[float] = None
    tick_size: Optional[float] = None
    expiration_cycle: Optional[str] = None  # "quarterly", "monthly", None
    requires_rollover: bool = False
    
    # Market data
    yfinance_symbol: Optional[str] = None
    trading_hours: str = "US_REGULAR"  # "US_REGULAR", "24H", "CUSTOM"
    
    # Risk parameters
    typical_margin_pct: Optional[float] = None  # For futures
    
    # Seasonality (for commodities)
    has_seasonality: bool = False
    seasonality_pattern: Optional[str] = None  # "winter_demand", "summer_driving", etc.


# =============================================================================
# ASSET REGISTRY - Single Source of Truth
# =============================================================================

ASSET_REGISTRY: Dict[str, AssetMetadata] = {
    
    # =========================================================================
    # EQUITY INDICES (Futures)
    # =========================================================================
    
    'ES': AssetMetadata(
        ticker='ES',
        name='S&P 500 E-mini Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.EQUITY,
        contract_multiplier=50,  # $50 per point
        tick_size=0.25,
        expiration_cycle='quarterly',  # Mar, Jun, Sep, Dec
        requires_rollover=True,
        yfinance_symbol='ES=F',
        trading_hours='24H',
        typical_margin_pct=0.05  # ~5% margin requirement
    ),
    
    'NQ': AssetMetadata(
        ticker='NQ',
        name='NASDAQ 100 E-mini Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.EQUITY,
        contract_multiplier=20,  # $20 per point
        tick_size=0.25,
        expiration_cycle='quarterly',
        requires_rollover=True,
        yfinance_symbol='NQ=F',
        trading_hours='24H',
        typical_margin_pct=0.06
    ),
    
    'RTY': AssetMetadata(
        ticker='RTY',
        name='Russell 2000 E-mini Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.EQUITY,
        contract_multiplier=50,  # $50 per point
        tick_size=0.10,
        expiration_cycle='quarterly',
        requires_rollover=True,
        yfinance_symbol='RTY=F',
        trading_hours='24H',
        typical_margin_pct=0.06
    ),
    
    'YM': AssetMetadata(
        ticker='YM',
        name='Dow Jones E-mini Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.EQUITY,
        contract_multiplier=5,  # $5 per point
        tick_size=1.0,
        expiration_cycle='quarterly',
        requires_rollover=True,
        yfinance_symbol='YM=F',
        trading_hours='24H',
        typical_margin_pct=0.05
    ),
    
    'MME': AssetMetadata(
        ticker='MME',
        name='Emerging markets Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.EQUITY,
        contract_multiplier=20,  # $20 per point
        tick_size=0.25,
        expiration_cycle='quarterly',
        requires_rollover=True,
        yfinance_symbol='MME=F',
        trading_hours='24H',
        typical_margin_pct=0.06
    ),
    
    'NIY': AssetMetadata(
        ticker='NIY',
        name='Japan Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.EQUITY,
        contract_multiplier=20,  # $20 per point
        tick_size=0.25,
        expiration_cycle='quarterly',
        requires_rollover=True,
        yfinance_symbol='NIY=F',
        trading_hours='24H',
        typical_margin_pct=0.06
    ),
    
    # =========================================================================
    # ENERGY COMMODITIES (Futures)
    # =========================================================================
    
    'CL': AssetMetadata(
        ticker='CL',
        name='WTI Crude Oil Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.COMMODITY_ENERGY,
        contract_multiplier=1000,  # 1,000 barrels per contract
        tick_size=0.01,  # $0.01 per barrel = $10 per contract
        expiration_cycle='monthly',
        requires_rollover=True,
        yfinance_symbol='CL=F',
        trading_hours='24H',
        typical_margin_pct=0.08,  # ~8% margin
        has_seasonality=True,
        seasonality_pattern='summer_driving'  # Demand peaks before summer driving season
    ),
    
    'NG': AssetMetadata(
        ticker='NG',
        name='Natural Gas Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.COMMODITY_ENERGY,
        contract_multiplier=10000,  # 10,000 MMBtu per contract
        tick_size=0.001,  # $0.001 per MMBtu = $10 per contract
        expiration_cycle='monthly',
        requires_rollover=True,
        yfinance_symbol='NG=F',
        trading_hours='24H',
        typical_margin_pct=0.10,  # ~10% margin (volatile)
        has_seasonality=True,
        seasonality_pattern='winter_demand'  # Heating demand peaks in winter
    ),
    
    'RB': AssetMetadata(
        ticker='RB',
        name='RBOB Gasoline Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.COMMODITY_ENERGY,
        contract_multiplier=42000,  # 42,000 gallons per contract (1,000 barrels)
        tick_size=0.0001,  # $0.0001 per gallon
        expiration_cycle='monthly',
        requires_rollover=True,
        yfinance_symbol='RB=F',
        trading_hours='24H',
        typical_margin_pct=0.08,
        has_seasonality=True,
        seasonality_pattern='summer_driving'
    ),
    
    'HO': AssetMetadata(
        ticker='HO',
        name='Heating Oil Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.COMMODITY_ENERGY,
        contract_multiplier=42000,  # 42,000 gallons per contract
        tick_size=0.0001,
        expiration_cycle='monthly',
        requires_rollover=True,
        yfinance_symbol='HO=F',
        trading_hours='24H',
        typical_margin_pct=0.08,
        has_seasonality=True,
        seasonality_pattern='winter_demand'
    ),
    
    # =========================================================================
    # METALS (Futures)
    # =========================================================================
    
    'GC': AssetMetadata(
        ticker='GC',
        name='Gold Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.COMMODITY_METAL,
        contract_multiplier=100,  # 100 troy ounces per contract
        tick_size=0.10,  # $0.10 per ounce = $10 per contract
        expiration_cycle='monthly',
        requires_rollover=True,
        yfinance_symbol='GC=F',
        trading_hours='24H',
        typical_margin_pct=0.04  # ~4% margin (lower vol)
    ),
    
    'SI': AssetMetadata(
        ticker='SI',
        name='Silver Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.COMMODITY_METAL,
        contract_multiplier=5000,  # 5,000 troy ounces per contract
        tick_size=0.005,  # $0.005 per ounce = $25 per contract
        expiration_cycle='monthly',
        requires_rollover=True,
        yfinance_symbol='SI=F',
        trading_hours='24H',
        typical_margin_pct=0.06
    ),
    
    'HG': AssetMetadata(
        ticker='HG',
        name='Copper Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.COMMODITY_METAL,
        contract_multiplier=25000,  # 25,000 pounds per contract
        tick_size=0.0005,  # $0.0005 per pound = $12.50 per contract
        expiration_cycle='monthly',
        requires_rollover=True,
        yfinance_symbol='HG=F',
        trading_hours='24H',
        typical_margin_pct=0.06
    ),
    
    'PL': AssetMetadata(
        ticker='PL',
        name='Platinum Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.COMMODITY_METAL,
        contract_multiplier=50,  # 50 troy ounces per contract
        tick_size=0.10,
        expiration_cycle='quarterly',
        requires_rollover=True,
        yfinance_symbol='PL=F',
        trading_hours='24H',
        typical_margin_pct=0.05
    ),
    
    # =========================================================================
    # AGRICULTURAL COMMODITIES (Futures)
    # =========================================================================
    
    'ZC': AssetMetadata(
        ticker='ZC',
        name='Corn Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.COMMODITY_AGRICULTURE,
        contract_multiplier=5000,  # 5,000 bushels per contract
        tick_size=0.0025,  # $0.25 per contract (quarter cent per bushel)
        expiration_cycle='monthly',
        requires_rollover=True,
        yfinance_symbol='ZC=F',
        trading_hours='US_REGULAR',
        typical_margin_pct=0.05,
        has_seasonality=True,
        seasonality_pattern='harvest_cycle'  # Plant in spring, harvest in fall
    ),
    
    'ZS': AssetMetadata(
        ticker='ZS',
        name='Soybean Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.COMMODITY_AGRICULTURE,
        contract_multiplier=5000,  # 5,000 bushels per contract
        tick_size=0.0025,
        expiration_cycle='monthly',
        requires_rollover=True,
        yfinance_symbol='ZS=F',
        trading_hours='US_REGULAR',
        typical_margin_pct=0.05,
        has_seasonality=True,
        seasonality_pattern='harvest_cycle'
    ),
    
    'ZW': AssetMetadata(
        ticker='ZW',
        name='Wheat Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.COMMODITY_AGRICULTURE,
        contract_multiplier=5000,  # 5,000 bushels per contract
        tick_size=0.0025,
        expiration_cycle='monthly',
        requires_rollover=True,
        yfinance_symbol='ZW=F',
        trading_hours='US_REGULAR',
        typical_margin_pct=0.05,
        has_seasonality=True,
        seasonality_pattern='harvest_cycle'
    ),
    
    # =========================================================================
    # FIXED INCOME (Futures)
    # =========================================================================
    
    'ZN': AssetMetadata(
        ticker='ZN',
        name='10-Year Treasury Note Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.FIXED_INCOME,
        contract_multiplier=1000,  # $1,000 per point
        tick_size=0.015625,  # 1/64 of a point
        expiration_cycle='quarterly',
        requires_rollover=True,
        yfinance_symbol='ZN=F',
        trading_hours='24H',
        typical_margin_pct=0.02
    ),
    
    'ZB': AssetMetadata(
        ticker='ZB',
        name='30-Year Treasury Bond Futures',
        asset_type=AssetType.FUTURES,
        asset_class=AssetClass.FIXED_INCOME,
        contract_multiplier=1000,
        tick_size=0.03125,  # 1/32 of a point
        expiration_cycle='quarterly',
        requires_rollover=True,
        yfinance_symbol='ZB=F',
        trading_hours='24H',
        typical_margin_pct=0.03
    ),
    
    # =========================================================================
    # STOCKS & ETFs (Capital Markets)
    # =========================================================================
    
    'SPY': AssetMetadata(
        ticker='SPY',
        name='SPDR S&P 500 ETF',
        asset_type=AssetType.ETF,
        asset_class=AssetClass.EQUITY,
        requires_rollover=False,  # No rollover for ETFs
        yfinance_symbol='SPY',
        trading_hours='US_REGULAR'
    ),
    
    'QQQ': AssetMetadata(
        ticker='QQQ',
        name='Invesco QQQ Trust',
        asset_type=AssetType.ETF,
        asset_class=AssetClass.EQUITY,
        requires_rollover=False,
        yfinance_symbol='QQQ',
        trading_hours='US_REGULAR'
    ),
    
    'IWM': AssetMetadata(
        ticker='IWM',
        name='iShares Russell 2000 ETF',
        asset_type=AssetType.ETF,
        asset_class=AssetClass.EQUITY,
        requires_rollover=False,
        yfinance_symbol='IWM',
        trading_hours='US_REGULAR'
    ),
    
    'TLT': AssetMetadata(
        ticker='TLT',
        name='iShares 20+ Year Treasury ETF',
        asset_type=AssetType.ETF,
        asset_class=AssetClass.FIXED_INCOME,
        requires_rollover=False,
        yfinance_symbol='TLT',
        trading_hours='US_REGULAR'
    ),
    
    'GLD': AssetMetadata(
        ticker='GLD',
        name='SPDR Gold Trust',
        asset_type=AssetType.ETF,
        asset_class=AssetClass.COMMODITY_METAL,
        requires_rollover=False,
        yfinance_symbol='GLD',
        trading_hours='US_REGULAR'
    ),
    
    'USO': AssetMetadata(
        ticker='USO',
        name='United States Oil Fund',
        asset_type=AssetType.ETF,
        asset_class=AssetClass.COMMODITY_ENERGY,
        requires_rollover=False,
        yfinance_symbol='USO',
        trading_hours='US_REGULAR'
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_asset(ticker: str) -> AssetMetadata:
    """
    Get asset metadata by ticker symbol.
    
    Args:
        ticker: Asset ticker (e.g., 'CL', 'ES', 'SPY')
    
    Returns:
        AssetMetadata object
    
    Raises:
        KeyError: If ticker not found in registry
    
    Example:
        >>> metadata = get_asset('CL')
        >>> print(f"{metadata.name}: {metadata.contract_multiplier} barrels")
        WTI Crude Oil Futures: 1000 barrels
    """
    if ticker not in ASSET_REGISTRY:
        available = ', '.join(sorted(ASSET_REGISTRY.keys()))
        raise KeyError(
            f"Asset '{ticker}' not found in registry. "
            f"Available assets: {available}"
        )
    return ASSET_REGISTRY[ticker]


def filter_by_class(asset_class: AssetClass) -> List[AssetMetadata]:
    """
    Get all assets of a specific asset class.
    
    Args:
        asset_class: AssetClass enum value
    
    Returns:
        List of AssetMetadata objects
    
    Example:
        >>> energy_assets = filter_by_class(AssetClass.COMMODITY_ENERGY)
        >>> print([a.ticker for a in energy_assets])
        ['CL', 'NG', 'RB', 'HO']
    """
    return [
        metadata for metadata in ASSET_REGISTRY.values()
        if metadata.asset_class == asset_class
    ]


def filter_by_type(asset_type: AssetType) -> List[AssetMetadata]:
    """
    Get all assets of a specific type.
    
    Args:
        asset_type: AssetType enum value
    
    Returns:
        List of AssetMetadata objects
    
    Example:
        >>> futures = filter_by_type(AssetType.FUTURES)
        >>> print(f"Total futures: {len(futures)}")
        Total futures: 18
    """
    return [
        metadata for metadata in ASSET_REGISTRY.values()
        if metadata.asset_type == asset_type
    ]


def get_futures_requiring_rollover() -> List[AssetMetadata]:
    """
    Get all assets that require rollover handling.
    
    Returns:
        List of AssetMetadata objects with requires_rollover=True
    
    Example:
        >>> rollover_assets = get_futures_requiring_rollover()
        >>> print([a.ticker for a in rollover_assets])
        ['ES', 'NQ', 'CL', 'NG', 'GC', ...]
    """
    return [
        metadata for metadata in ASSET_REGISTRY.values()
        if metadata.requires_rollover
    ]


def get_seasonal_commodities() -> List[AssetMetadata]:
    """
    Get all assets with seasonal patterns.
    
    Returns:
        List of AssetMetadata objects with has_seasonality=True
    
    Example:
        >>> seasonal = get_seasonal_commodities()
        >>> for asset in seasonal:
        ...     print(f"{asset.ticker}: {asset.seasonality_pattern}")
        CL: summer_driving
        NG: winter_demand
        ZC: harvest_cycle
    """
    return [
        metadata for metadata in ASSET_REGISTRY.values()
        if metadata.has_seasonality
    ]


def get_all_tickers() -> List[str]:
    """
    Get list of all available ticker symbols.
    
    Returns:
        Sorted list of ticker strings
    
    Example:
        >>> tickers = get_all_tickers()
        >>> print(f"Total assets: {len(tickers)}")
        Total assets: 25
    """
    return sorted(ASSET_REGISTRY.keys())


def print_registry_summary():
    """
    Print summary statistics of the asset registry.
    
    Useful for debugging and validation.
    """
    print("=" * 80)
    print("ASSET REGISTRY SUMMARY")
    print("=" * 80)
    
    # Count by type
    type_counts = {}
    for asset_type in AssetType:
        count = len(filter_by_type(asset_type))
        if count > 0:
            type_counts[asset_type.value] = count
    
    print("\nBy Asset Type:")
    for asset_type, count in sorted(type_counts.items()):
        print(f"  {asset_type:15s}: {count:3d} assets")
    
    # Count by class
    class_counts = {}
    for asset_class in AssetClass:
        count = len(filter_by_class(asset_class))
        if count > 0:
            class_counts[asset_class.value] = count
    
    print("\nBy Asset Class:")
    for asset_class, count in sorted(class_counts.items()):
        print(f"  {asset_class:25s}: {count:3d} assets")
    
    # Special categories
    rollover_count = len(get_futures_requiring_rollover())
    seasonal_count = len(get_seasonal_commodities())
    
    print(f"\nSpecial Categories:")
    print(f"  Requires rollover:        {rollover_count:3d} assets")
    print(f"  Has seasonality:          {seasonal_count:3d} assets")
    
    print(f"\nTotal Assets:               {len(ASSET_REGISTRY):3d}")
    print("=" * 80)


if __name__ == "__main__":
    # Demo usage
    print_registry_summary()
    
    print("\n" + "=" * 80)
    print("EXAMPLE QUERIES")
    print("=" * 80)
    
    # Example 1: Get specific asset
    print("\n1. Get CL metadata:")
    cl = get_asset('CL')
    print(f"   Name: {cl.name}")
    print(f"   Contract size: {cl.contract_multiplier} barrels")
    print(f"   Margin: {cl.typical_margin_pct:.1%}")
    print(f"   Seasonality: {cl.seasonality_pattern}")
    
    # Example 2: Filter by class
    print("\n2. Energy commodities:")
    energy = filter_by_class(AssetClass.COMMODITY_ENERGY)
    for asset in energy:
        print(f"   {asset.ticker:5s}: {asset.name}")
    
    # Example 3: Futures requiring rollover
    print("\n3. Futures requiring rollover:")
    rollover_assets = get_futures_requiring_rollover()
    tickers = [a.ticker for a in rollover_assets]
    print(f"   {', '.join(tickers)}")
    
    # Example 4: Seasonal commodities
    print("\n4. Commodities with seasonal patterns:")
    seasonal = get_seasonal_commodities()
    for asset in seasonal:
        print(f"   {asset.ticker:5s}: {asset.seasonality_pattern}")
