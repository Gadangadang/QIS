"""
Example TAA analysis script with FactSet support.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.taa.features.pipeline import FeaturePipeline
from core.benchmark import BenchmarkLoader
from core.data.collectors import FACTSET_AVAILABLE

# Determine data source
USE_FACTSET = FACTSET_AVAILABLE  # Auto-detect FactSet environment

print(f"Using data source: {'FactSet' if USE_FACTSET else 'Yahoo Finance'}")

# Initialize pipeline
pipeline = FeaturePipeline(use_factset=USE_FACTSET)

# Run feature generation
tickers = ['SPY', 'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU']
features = pipeline.run(
    tickers=tickers,
    benchmark_ticker='SPY',
    start_date='2020-01-01',
    end_date='2024-01-01'
)

print(f"Features shape: {features.shape}")
print(features.head())

# Load benchmark for comparison
benchmark_loader = BenchmarkLoader(use_factset=USE_FACTSET)
benchmark = benchmark_loader.load_benchmark(
    ticker='SPY',
    start_date='2020-01-01',
    end_date='2024-01-01'
)

print(f"Benchmark shape: {benchmark.shape}")
print(benchmark.head())