"""
FRED Data Collector for Economic Indicators.

Fetches macroeconomic data from Federal Reserve Economic Data (FRED).
Supports growth, inflation, policy, and sentiment indicators for TAA regime detection.
"""

from typing import List, Optional, Union, Dict
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import logging
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class FredCollector(BaseCollector):
    """
    Data collector for Federal Reserve Economic Data (FRED).
    
    Provides access to key macro indicators for TAA:
    - Growth: GDP, PMI, Employment
    - Inflation: CPI, PCE, Breakevens
    - Policy: Fed Funds Rate, Treasury Yields
    - Risk: Credit Spreads, VIX
    
    Example:
        >>> collector = FredCollector()
        >>> growth = collector.fetch_growth_indicators(start_date='2010-01-01')
        >>> cpi = collector.fetch_history(['CPIAUCSL'], start_date='2020-01-01')
    """
    
    # Common FRED series IDs for TAA
    SERIES_MAP = {
        # Growth Indicators
        'gdp': 'GDP',                              # Gross Domestic Product
        'gdp_growth': 'A191RL1Q225SBEA',          # Real GDP Growth Rate
        'unemployment': 'UNRATE',                  # Unemployment Rate
        'payrolls': 'PAYEMS',                      # Nonfarm Payrolls
        'retail_sales': 'RSXFS',                   # Retail Sales
        'industrial_prod': 'INDPRO',               # Industrial Production
        'capacity_util': 'TCU',                    # Capacity Utilization
        
        # Inflation Indicators
        'cpi': 'CPIAUCSL',                         # Consumer Price Index
        'cpi_core': 'CPILFESL',                    # Core CPI (ex food/energy)
        'pce': 'PCE',                              # Personal Consumption Expenditures
        'pce_core': 'PCEPILFE',                    # Core PCE
        'ppi': 'PPIACO',                           # Producer Price Index
        'inflation_expectations': 'T5YIE',         # 5Y Breakeven Inflation
        
        # Policy & Rates
        'fed_funds': 'DFF',                        # Federal Funds Rate
        'treasury_10y': 'DGS10',                   # 10-Year Treasury Yield
        'treasury_2y': 'DGS2',                     # 2-Year Treasury Yield
        'treasury_3m': 'DGS3MO',                   # 3-Month Treasury Yield
        'treasury_1y': 'DGS1',                     # 1-Year Treasury Yield
        'treasury_5y': 'DGS5',                     # 5-Year Treasury Yield
        'mortgage_30y': 'MORTGAGE30US',            # 30-Year Mortgage Rate
        
        # Credit & Risk
        'baa_spread': 'BAA10Y',                    # BAA Corporate Spread
        'aaa_spread': 'AAA10Y',                    # AAA Corporate Spread
        'ted_spread': 'TEDRATE',                   # TED Spread
        'vix': 'VIXCLS',                           # VIX (via FRED)
        
        # Money Supply & Liquidity
        'm2': 'M2SL',                              # M2 Money Supply
        'm1': 'M1SL',                              # M1 Money Supply
    }

    def fetch_history(
        self, 
        tickers: List[str], 
        start_date: Union[str, datetime], 
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data from FRED.
        
        Args:
            tickers: List of FRED series IDs (e.g., ['GDP', 'UNRATE']) or friendly names (e.g., ['gdp', 'unemployment']).
            start_date: Start date.
            end_date: End date.
            interval: Not used for FRED (frequency is determined by the series).
            
        Returns:
            pd.DataFrame: DataFrame with columns as Series IDs.
            
        Example:
            >>> collector = FredCollector()
            >>> df = collector.fetch_history(['gdp', 'unemployment'], start_date='2020-01-01')
        """
        if not tickers:
            logger.warning("No tickers provided to fetch_history")
            return pd.DataFrame()
        
        # Map friendly names to FRED series IDs
        fred_ids = []
        for ticker in tickers:
            if ticker.lower() in self.SERIES_MAP:
                fred_ids.append(self.SERIES_MAP[ticker.lower()])
            else:
                fred_ids.append(ticker)

        logger.info(f"Fetching FRED data for {len(fred_ids)} series from {start_date} to {end_date}")
        
        try:
            # pandas_datareader handles multiple tickers by returning a DataFrame with columns=tickers
            df = web.DataReader(fred_ids, 'fred', start_date, end_date)
            
            if df.empty:
                logger.warning("FRED returned empty DataFrame")
                return df
            
            # Forward fill missing values (different series have different frequencies)
            df = df.ffill()
                
            logger.info(f"Retrieved {len(df)} observations for {len(fred_ids)} series")
            return df

        except Exception as e:
            logger.error(f"Error fetching data from FRED: {e}")
            raise

    def fetch_latest(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch latest data point.
        """
        # FRED data is often monthly/quarterly, so we need a longer lookback to find the "latest"
        end = datetime.now()
        start = end - pd.Timedelta(days=90) # Look back 3 months to be safe
        
        df = self.fetch_history(tickers, start_date=start, end_date=end)
        
        if df.empty:
            return df
            
        return df.iloc[[-1]]
    
    def fetch_growth_indicators(
        self,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Fetch standard growth indicators for regime detection.
        
        Args:
            start_date: Start date
            end_date: End date (default: latest)
        
        Returns:
            DataFrame with GDP growth, unemployment, payrolls, industrial production
            
        Example:
            >>> collector = FredCollector()
            >>> growth = collector.fetch_growth_indicators(start_date='2010-01-01')
        """
        indicators = ['gdp_growth', 'unemployment', 'payrolls', 'industrial_prod']
        return self.fetch_history(indicators, start_date, end_date)
    
    def fetch_inflation_indicators(
        self,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Fetch standard inflation indicators.
        
        Args:
            start_date: Start date
            end_date: End date (default: latest)
        
        Returns:
            DataFrame with CPI, core CPI, core PCE, inflation expectations
        """
        indicators = ['cpi', 'cpi_core', 'pce_core', 'inflation_expectations']
        return self.fetch_history(indicators, start_date, end_date)
    
    def fetch_policy_indicators(
        self,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Fetch policy and rate indicators.
        
        Args:
            start_date: Start date
            end_date: End date (default: latest)
        
        Returns:
            DataFrame with Fed Funds, Treasury yields, yield curve slope
        """
        indicators = ['fed_funds', 'treasury_10y', 'treasury_2y', 'treasury_3m']
        df = self.fetch_history(indicators, start_date, end_date)
        
        # Calculate yield curve slope if both yields available
        if 'DGS10' in df.columns and 'DGS2' in df.columns:
            df['yield_curve_slope'] = df['DGS10'] - df['DGS2']
        
        return df
    
    def fetch_risk_indicators(
        self,
        start_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Fetch risk and credit indicators.
        
        Args:
            start_date: Start date
            end_date: End date (default: latest)
        
        Returns:
            DataFrame with credit spreads, VIX, TED spread
        """
        indicators = ['baa_spread', 'ted_spread', 'vix']
        return self.fetch_history(indicators, start_date, end_date)
    
    @staticmethod
    def list_available_series() -> Dict[str, str]:
        """
        List all available FRED series with friendly names.
        
        Returns:
            Dict mapping friendly names to FRED series IDs
            
        Example:
            >>> series = FredCollector.list_available_series()
            >>> print(series['gdp'])  # Output: 'GDP'
        """
        return FredCollector.SERIES_MAP.copy()
