"""
Feature Pipeline.
Orchestrates data collection, alignment, and feature generation.
"""

import pandas as pd
import logging
from typing import List, Dict, Optional

# Import both collectors
from core.data.collectors import (
    YahooCollector, 
    FactSetCollector, 
    FACTSET_AVAILABLE
)
from core.data.processors import PriceProcessor
from .price import PriceFeatureGenerator
from .relative import RelativeValueFeatureGenerator

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    Orchestrates the end-to-end feature generation process.
    """
    
    def __init__(self, use_factset: bool = False):
        """
        Initialize feature pipeline.
        
        Args:
            use_factset: If True and available, use FactSet instead of Yahoo
        """
        # Choose data source
        if use_factset and FACTSET_AVAILABLE:
            logger.info("Using FactSet data collector")
            self.data_collector = FactSetCollector()
            self.use_factset = True
        else:
            if use_factset and not FACTSET_AVAILABLE:
                logger.warning("FactSet requested but not available, falling back to Yahoo")
            logger.info("Using Yahoo Finance data collector")
            self.data_collector = YahooCollector()
            self.use_factset = False
        
        self.processor = PriceProcessor()
        
        self.price_gen = PriceFeatureGenerator()
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
            benchmark_ticker: Benchmark ticker (e.g., 'ACWI', 'SPY').
            start_date: Start date.
            end_date: End date.
            
        Returns:
            pd.DataFrame: Master feature matrix (Date, Ticker) index.
        """
        logger.info("Starting Feature Pipeline...")
        
        # 1. Fetch Data
        logger.info("Fetching Asset Data...")
        assets_df = self.data_collector.fetch_history(tickers, start_date, end_date)
        assets_df = self.processor.process(assets_df)
        
        logger.info("Fetching Benchmark Data...")
        bench_df = self.data_collector.fetch_history([benchmark_ticker], start_date, end_date)
        bench_df = self.processor.process(bench_df)
        
        # 2. Generate Features
        logger.info("Generating Price Features...")
        price_feats = self.price_gen.generate(assets_df)
        
        logger.info("Generating Relative Features...")
        rel_feats = self.rel_gen.generate(assets_df, benchmark=bench_df)
        
        # 3. Merge Everything
        logger.info("Merging Features...")
        
        master_df = pd.merge(
            price_feats, 
            rel_feats, 
            left_index=True, 
            right_index=True, 
            how='inner'
        )
        
        # Set index to (Date, Ticker)
        master_df = master_df.sort_index()
        
        logger.info(f"Feature pipeline complete. Shape: {master_df.shape}")
        return master_df