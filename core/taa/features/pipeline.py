"""
Feature Pipeline.
Orchestrates data collection, alignment, and feature generation.
"""

import pandas as pd
import logging
from typing import List, Dict, Optional
from core.data.collectors import YahooCollector, FredCollector
from core.data.processors import PriceProcessor
from .price import PriceFeatureGenerator
from .macro import MacroFeatureGenerator
from .relative import RelativeValueFeatureGenerator

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    Orchestrates the end-to-end feature generation process.
    """
    
    def __init__(self):
        self.yahoo = YahooCollector()
        self.fred = FredCollector()
        self.processor = PriceProcessor()
        
        self.price_gen = PriceFeatureGenerator()
        self.macro_gen = MacroFeatureGenerator()
        self.rel_gen = RelativeValueFeatureGenerator()

    def run(self, 
            tickers: List[str], 
            benchmark_ticker: str, 
            start_date: str, 
            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Run the pipeline.
        
        Args:
            tickers: List of asset tickers (e.g., Sector ETFs).
            benchmark_ticker: Benchmark ticker (e.g., 'ACWI').
            start_date: Start date.
            end_date: End date.
            
        Returns:
            pd.DataFrame: Master feature matrix (Date, Ticker) index.
        """
        logger.info("Starting Feature Pipeline...")
        
        # 1. Fetch Data
        logger.info("Fetching Asset Data...")
        assets_df = self.yahoo.fetch_history(tickers, start_date, end_date)
        assets_df = self.processor.process(assets_df)
        
        logger.info("Fetching Benchmark Data...")
        bench_df = self.yahoo.fetch_history([benchmark_ticker], start_date, end_date)
        bench_df = self.processor.process(bench_df)
        
        logger.info("Fetching Macro Data...")
        # Standard FRED IDs for TAA
        fred_tickers = ['T10Y2Y', 'BAA10Y', 'VIXCLS', 'CPIAUCSL']
        macro_raw = self.fred.fetch_history(fred_tickers, start_date, end_date)
        # Macro data needs forward filling to align with daily trading days
        macro_raw = macro_raw.ffill()
        
        # 2. Generate Features
        logger.info("Generating Price Features...")
        price_feats = self.price_gen.generate(assets_df)
        
        logger.info("Generating Macro Features...")
        macro_feats = self.macro_gen.generate(macro_raw)
        
        logger.info("Generating Relative Features...")
        rel_feats = self.rel_gen.generate(assets_df, benchmark=bench_df)
        
        # 3. Merge Everything
        logger.info("Merging Features...")
        
        # Price and Relative features are indexed by (Date, Ticker)
        # Macro features are indexed by (Date) -> need to broadcast to all tickers
        
        # Merge Price and Relative first
        master_df = pd.merge(
            price_feats, 
            rel_feats, 
            left_index=True, 
            right_index=True, 
            how='inner'
        )
        
        # Merge Macro
        # Reset index to merge on Date
        master_df = master_df.reset_index()
        macro_feats = macro_feats.reset_index().rename(columns={'index': 'Date', 'DATE': 'Date'})
        
        # Ensure Date columns are datetime
        master_df['Date'] = pd.to_datetime(master_df['Date'])
        macro_feats['Date'] = pd.to_datetime(macro_feats['Date'])
        
        # Merge macro (left join to keep asset dates)
        master_df = pd.merge(master_df, macro_feats, on='Date', how='left')
        
        # Set index back to (Date, Ticker)
        master_df = master_df.set_index(['Date', 'ticker']).sort_index()
        
        # Forward fill any missing macro data that might have occurred due to date mismatch
        master_df = master_df.groupby(level='ticker').ffill()
        
        logger.info(f"Feature Pipeline Complete. Shape: {master_df.shape}")
        return master_df
