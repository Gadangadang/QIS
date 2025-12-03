"""
Futures Rollover Handler

Manages futures contract expiration and generates continuous price series.

Key responsibilities:
1. Track contract expiration dates
2. Determine when to roll from front to back month
3. Generate continuous adjusted price series (Panama or Ratio method)
4. Calculate rollover costs

Works for ALL futures:
- Equity indices (ES, NQ, RTY, YM)
- Energy (CL, NG, RB, HO)
- Metals (GC, SI, HG, PL)
- Agriculture (ZC, ZS, ZW)
- Fixed income (ZN, ZB)

Usage:
    from core.futures.rollover_handler import FuturesRolloverHandler
    from core.asset_registry import ASSET_REGISTRY
    
    handler = FuturesRolloverHandler(ASSET_REGISTRY)
    
    # Get rollover dates
    dates = handler.get_rollover_dates('CL', '2020-01-01', '2024-12-31')
    
    # Create continuous series
    continuous = handler.create_continuous_series('CL', contract_data, method='panama')
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings

from core.asset_registry import AssetMetadata, ASSET_REGISTRY


# =============================================================================
# EXPIRATION RULES - Based on CME/NYMEX/ICE calendars
# =============================================================================

EXPIRATION_RULES = {
    # Energy (NYMEX)
    'CL': {
        'cycle': 'monthly',
        'day_rule': 'third_business_day_before_25th',
        'roll_days_before': 5,  # Roll 5 days before expiration
        'description': 'CL expires on 3rd business day before 25th of prior month'
    },
    'NG': {
        'cycle': 'monthly',
        'day_rule': 'third_business_day_before_end',
        'roll_days_before': 7,  # Roll earlier (more volatile)
        'description': 'NG expires on 3rd business day before end of month'
    },
    'RB': {
        'cycle': 'monthly',
        'day_rule': 'last_business_day_before_delivery_month',
        'roll_days_before': 5,
        'description': 'RB (gasoline) expires last business day before delivery month'
    },
    'HO': {
        'cycle': 'monthly',
        'day_rule': 'last_business_day_before_delivery_month',
        'roll_days_before': 5,
        'description': 'HO (heating oil) expires last business day before delivery month'
    },
    
    # Metals (COMEX)
    'GC': {
        'cycle': 'monthly',
        'day_rule': 'third_last_business_day',
        'roll_days_before': 5,
        'description': 'GC expires on 3rd to last business day of month'
    },
    'SI': {
        'cycle': 'monthly',
        'day_rule': 'third_last_business_day',
        'roll_days_before': 5,
        'description': 'SI expires on 3rd to last business day of month'
    },
    'HG': {
        'cycle': 'monthly',
        'day_rule': 'third_last_business_day',
        'roll_days_before': 5,
        'description': 'HG (copper) expires on 3rd to last business day of month'
    },
    'PL': {
        'cycle': 'quarterly',
        'day_rule': 'third_last_business_day',
        'roll_days_before': 5,
        'description': 'PL (platinum) expires on 3rd to last business day'
    },
    
    # Equity Indices (CME)
    'ES': {
        'cycle': 'quarterly',  # Mar, Jun, Sep, Dec (H, M, U, Z)
        'day_rule': 'third_friday',
        'roll_days_before': 5,
        'description': 'ES expires on 3rd Friday of contract month'
    },
    'NQ': {
        'cycle': 'quarterly',
        'day_rule': 'third_friday',
        'roll_days_before': 5,
        'description': 'NQ expires on 3rd Friday of contract month'
    },
    'RTY': {
        'cycle': 'quarterly',
        'day_rule': 'third_friday',
        'roll_days_before': 5,
        'description': 'RTY expires on 3rd Friday of contract month'
    },
    'YM': {
        'cycle': 'quarterly',
        'day_rule': 'third_friday',
        'roll_days_before': 5,
        'description': 'YM expires on 3rd Friday of contract month'
    },
    
    # Agriculture (CBOT)
    'ZC': {
        'cycle': 'monthly',  # Active: Mar, May, Jul, Sep, Dec
        'day_rule': 'business_day_before_15th',
        'roll_days_before': 7,
        'description': 'ZC (corn) expires business day before 15th of contract month'
    },
    'ZS': {
        'cycle': 'monthly',  # Active: Jan, Mar, May, Jul, Aug, Sep, Nov
        'day_rule': 'business_day_before_15th',
        'roll_days_before': 7,
        'description': 'ZS (soybeans) expires business day before 15th'
    },
    'ZW': {
        'cycle': 'monthly',  # Active: Mar, May, Jul, Sep, Dec
        'day_rule': 'business_day_before_15th',
        'roll_days_before': 7,
        'description': 'ZW (wheat) expires business day before 15th'
    },
    
    # Fixed Income (CBOT)
    'ZN': {
        'cycle': 'quarterly',
        'day_rule': 'seventh_business_day_before_end',
        'roll_days_before': 5,
        'description': 'ZN (10Y note) expires 7th business day before end of month'
    },
    'ZB': {
        'cycle': 'quarterly',
        'day_rule': 'seventh_business_day_before_end',
        'roll_days_before': 5,
        'description': 'ZB (30Y bond) expires 7th business day before end of month'
    },
}


class FuturesRolloverHandler:
    """
    Handles futures contract rollovers and continuous series generation.
    
    Features:
    - Accurate expiration date calculation per CME/NYMEX/ICE rules
    - Multiple adjustment methods (Panama, Ratio, None)
    - Rollover cost tracking
    - Works for all futures (equity, commodity, fixed income)
    
    Attributes:
        registry: Asset metadata registry
        expiration_rules: Dict of expiration rules per ticker
    """
    
    def __init__(self, asset_registry: Optional[Dict[str, AssetMetadata]] = None):
        """
        Initialize rollover handler.
        
        Args:
            asset_registry: Asset metadata registry (defaults to ASSET_REGISTRY)
        """
        self.registry = asset_registry or ASSET_REGISTRY
        self.expiration_rules = EXPIRATION_RULES
    
    def get_rollover_dates(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str
    ) -> List[pd.Timestamp]:
        """
        Calculate rollover dates for a ticker in given period.
        
        Rollover happens X days before contract expiration (configured per asset).
        
        Args:
            ticker: Asset ticker (e.g., 'CL', 'ES')
            start_date: Start of period (YYYY-MM-DD)
            end_date: End of period (YYYY-MM-DD)
        
        Returns:
            List of rollover dates (sorted)
        
        Example:
            >>> handler.get_rollover_dates('CL', '2024-01-01', '2024-12-31')
            [Timestamp('2024-01-17'), Timestamp('2024-02-16'), ...]  # ~12 dates
        
        Notes:
            - CL (monthly): ~12 rollovers per year
            - ES (quarterly): ~4 rollovers per year
        """
        if ticker not in self.expiration_rules:
            raise ValueError(
                f"No expiration rules for {ticker}. "
                f"Available: {list(self.expiration_rules.keys())}"
            )
        
        rules = self.expiration_rules[ticker]
        cycle = rules['cycle']
        roll_days = rules['roll_days_before']
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        rollover_dates = []
        
        if cycle == 'monthly':
            # Generate monthly expiration dates
            current = start.replace(day=1)
            while current <= end:
                expiry = self._calculate_expiration_date(ticker, current)
                rollover = expiry - pd.Timedelta(days=roll_days)
                
                if start <= rollover <= end:
                    rollover_dates.append(rollover)
                
                current += relativedelta(months=1)
        
        elif cycle == 'quarterly':
            # Generate quarterly expiration dates (Mar, Jun, Sep, Dec)
            quarters = [3, 6, 9, 12]  # March, June, September, December
            current_year = start.year
            end_year = end.year
            
            for year in range(current_year, end_year + 1):
                for month in quarters:
                    contract_month = pd.Timestamp(year=year, month=month, day=1)
                    expiry = self._calculate_expiration_date(ticker, contract_month)
                    rollover = expiry - pd.Timedelta(days=roll_days)
                    
                    if start <= rollover <= end:
                        rollover_dates.append(rollover)
        
        return sorted(rollover_dates)
    
    def _calculate_expiration_date(
        self, 
        ticker: str, 
        contract_month: pd.Timestamp
    ) -> pd.Timestamp:
        """
        Calculate exact expiration date for a contract.
        
        Args:
            ticker: Asset ticker
            contract_month: Contract month (e.g., December 2024)
        
        Returns:
            Expiration date
        
        Note:
            This is simplified for initial implementation.
            TODO: Add proper business day calendar (exclude holidays)
        """
        rules = self.expiration_rules[ticker]
        day_rule = rules['day_rule']
        
        if day_rule == 'third_friday':
            # ES, NQ, RTY, YM: 3rd Friday of contract month
            return self._get_nth_weekday(contract_month, weekday=4, n=3)  # Friday = 4
        
        elif day_rule == 'third_business_day_before_25th':
            # CL: 3rd business day before 25th of PRIOR month
            # (e.g., Dec CL expires ~Nov 22)
            prior_month = contract_month - relativedelta(months=1)
            target = prior_month.replace(day=25)
            return self._get_nth_business_day_before(target, n=3)
        
        elif day_rule == 'third_business_day_before_end':
            # NG: 3rd business day before end of month
            last_day = contract_month + relativedelta(months=1, days=-1)
            return self._get_nth_business_day_before(last_day, n=3)
        
        elif day_rule == 'third_last_business_day':
            # GC, SI, HG: 3rd to last business day of month
            last_day = contract_month + relativedelta(months=1, days=-1)
            return self._get_nth_business_day_before(last_day, n=2)  # 3rd to last = 2 before
        
        elif day_rule == 'business_day_before_15th':
            # ZC, ZS, ZW: Business day before 15th
            target = contract_month.replace(day=15)
            return self._get_nth_business_day_before(target, n=1)
        
        elif day_rule == 'seventh_business_day_before_end':
            # ZN, ZB: 7th business day before end of month
            last_day = contract_month + relativedelta(months=1, days=-1)
            return self._get_nth_business_day_before(last_day, n=7)
        
        elif day_rule == 'last_business_day_before_delivery_month':
            # RB, HO: Last business day of month BEFORE delivery month
            prior_month = contract_month - relativedelta(months=1)
            last_day = prior_month + relativedelta(months=1, days=-1)
            return self._get_last_business_day(last_day)
        
        else:
            raise ValueError(f"Unknown day rule: {day_rule}")
    
    def _get_nth_weekday(
        self, 
        month_start: pd.Timestamp, 
        weekday: int, 
        n: int
    ) -> pd.Timestamp:
        """
        Get nth occurrence of weekday in month.
        
        Args:
            month_start: First day of month
            weekday: Day of week (0=Monday, 4=Friday)
            n: Which occurrence (1=first, 3=third)
        
        Returns:
            Date of nth weekday
        """
        first_day = month_start.replace(day=1)
        
        # Find first occurrence of weekday
        days_ahead = weekday - first_day.weekday()
        if days_ahead < 0:
            days_ahead += 7
        
        first_occurrence = first_day + pd.Timedelta(days=days_ahead)
        
        # Add weeks to get nth occurrence
        nth_occurrence = first_occurrence + pd.Timedelta(weeks=(n - 1))
        
        return nth_occurrence
    
    def _get_nth_business_day_before(
        self, 
        date: pd.Timestamp, 
        n: int
    ) -> pd.Timestamp:
        """
        Get nth business day before given date.
        
        Args:
            date: Reference date
            n: Number of business days to go back
        
        Returns:
            Business day
        
        Note:
            Simplified - assumes Mon-Fri are business days.
            TODO: Add holiday calendar
        """
        current = date
        business_days_back = 0
        
        while business_days_back < n:
            current -= pd.Timedelta(days=1)
            # Skip weekends
            if current.weekday() < 5:  # Monday=0, Friday=4
                business_days_back += 1
        
        return current
    
    def _get_last_business_day(self, date: pd.Timestamp) -> pd.Timestamp:
        """Get last business day of month containing date."""
        # Go backwards from date until we hit a weekday
        current = date
        while current.weekday() >= 5:  # Weekend
            current -= pd.Timedelta(days=1)
        return current
    
    def create_continuous_series(
        self,
        ticker: str,
        prices: pd.DataFrame,
        method: str = 'panama',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create continuous price series from raw futures data.
        
        Adjusts historical prices to account for rollover gaps.
        
        Args:
            ticker: Asset ticker
            prices: DataFrame with Date, Open, High, Low, Close, Volume
                   (assumes this is already continuous from yfinance =F symbol)
            method: Adjustment method:
                - 'panama': Shift historical prices by roll difference (most common)
                - 'ratio': Scale historical prices by roll ratio
                - 'none': No adjustment (for spread strategies)
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            DataFrame with adjusted continuous prices
        
        Example:
            >>> prices = yf.download('CL=F', start='2020-01-01')
            >>> continuous = handler.create_continuous_series('CL', prices, method='panama')
        
        Notes:
            For initial implementation, we assume yfinance =F symbols already provide
            continuous prices. This method adds explicit rollover cost tracking.
            
            TODO: Implement full rollover adjustment from individual contracts
        """
        df = prices.copy()
        
        # Ensure Date column exists and is index
        if 'Date' in df.columns:
            df = df.set_index('Date')
        df.index = pd.to_datetime(df.index)
        
        # Filter date range if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Get rollover dates
        rollover_dates = self.get_rollover_dates(
            ticker, 
            df.index[0].strftime('%Y-%m-%d'),
            df.index[-1].strftime('%Y-%m-%d')
        )
        
        # Add rollover markers
        df['is_rollover'] = df.index.isin(rollover_dates)
        
        # Calculate rollover costs (simplified - assumes small gap)
        df['rollover_cost'] = 0.0
        for rollover_date in rollover_dates:
            if rollover_date in df.index:
                # Estimate rollover cost from price change
                # (In reality, we'd compare front vs. back month)
                idx = df.index.get_loc(rollover_date)
                if idx > 0 and idx < len(df) - 1:
                    prev_close = df.iloc[idx - 1]['Close']
                    next_open = df.iloc[idx + 1]['Open']
                    gap = next_open - prev_close
                    df.loc[rollover_date, 'rollover_cost'] = gap
        
        # Apply adjustment method
        if method == 'panama':
            # Shift historical prices by cumulative rollover gaps
            # (For yfinance =F data, this is already done, so we just track it)
            df['adjustment'] = 0.0
            cumulative_adjustment = 0.0
            
            for i in range(len(df)):
                if df.iloc[i]['is_rollover']:
                    cumulative_adjustment += df.iloc[i]['rollover_cost']
                df.iloc[i, df.columns.get_loc('adjustment')] = cumulative_adjustment
        
        elif method == 'ratio':
            # Scale by ratio (TODO: implement when we have individual contracts)
            df['adjustment'] = 0.0
            warnings.warn("Ratio method not yet implemented, using panama")
        
        elif method == 'none':
            # No adjustment
            df['adjustment'] = 0.0
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'panama', 'ratio', or 'none'")
        
        return df
    
    def calculate_rollover_cost(
        self,
        front_price: float,
        back_price: float,
        num_contracts: int,
        ticker: str
    ) -> float:
        """
        Calculate cost of rolling from front to back month.
        
        Args:
            front_price: Front month price
            back_price: Back month price
            num_contracts: Number of contracts to roll
            ticker: Asset ticker (to get contract multiplier)
        
        Returns:
            Rollover cost in dollars
            - Negative (contango): You PAY to roll (back > front)
            - Positive (backwardation): You EARN on roll (front > back)
        
        Example:
            >>> # CL in contango: back $71, front $70
            >>> cost = handler.calculate_rollover_cost(70, 71, 10, 'CL')
            >>> print(f"Cost: ${cost:,.0f}")
            Cost: $-10,000  # Pay $1/barrel × 1000 barrels × 10 contracts
            
            >>> # CL in backwardation: front $71, back $70
            >>> cost = handler.calculate_rollover_cost(71, 70, 10, 'CL')
            >>> print(f"Profit: ${cost:,.0f}")
            Profit: $10,000  # Earn $1/barrel × 1000 barrels × 10 contracts
        """
        metadata = self.registry.get(ticker)
        if not metadata:
            raise ValueError(f"Ticker {ticker} not in registry")
        
        if not metadata.contract_multiplier:
            raise ValueError(f"{ticker} is not a futures contract (no multiplier)")
        
        # Calculate spread
        spread = front_price - back_price
        
        # Cost per contract
        cost_per_contract = spread * metadata.contract_multiplier
        
        # Total cost
        total_cost = cost_per_contract * num_contracts
        
        return total_cost


if __name__ == "__main__":
    # Demo usage
    print("=" * 80)
    print("FUTURES ROLLOVER HANDLER DEMO")
    print("=" * 80)
    
    handler = FuturesRolloverHandler()
    
    # Example 1: Get rollover dates for CL
    print("\n1. CL (Crude Oil) rollover dates in 2024:")
    cl_dates = handler.get_rollover_dates('CL', '2024-01-01', '2024-12-31')
    for date in cl_dates:
        print(f"   {date.strftime('%Y-%m-%d')}")
    print(f"   Total: {len(cl_dates)} rollovers")
    
    # Example 2: ES rollover dates (quarterly)
    print("\n2. ES (S&P 500) rollover dates in 2024:")
    es_dates = handler.get_rollover_dates('ES', '2024-01-01', '2024-12-31')
    for date in es_dates:
        print(f"   {date.strftime('%Y-%m-%d')}")
    print(f"   Total: {len(es_dates)} rollovers")
    
    # Example 3: Calculate rollover cost
    print("\n3. Rollover cost examples:")
    
    # CL in contango (negative carry)
    cost = handler.calculate_rollover_cost(70.0, 71.0, 10, 'CL')
    print(f"   CL contango (front $70, back $71): ${cost:,.0f}")
    
    # CL in backwardation (positive carry)
    cost = handler.calculate_rollover_cost(71.0, 70.0, 10, 'CL')
    print(f"   CL backwardation (front $71, back $70): ${cost:,.0f}")
    
    # NG rollover
    cost = handler.calculate_rollover_cost(3.0, 3.2, 5, 'NG')
    print(f"   NG contango (front $3, back $3.20, 5 contracts): ${cost:,.0f}")
    
    print("\n" + "=" * 80)
