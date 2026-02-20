"""
Data collectors module.
"""

from .yahoo_collector import YahooCollector

# Only import FactSet if running in FactSet environment
try:
    from .factset_collector import FactSetCollector, fetch_benchmark, BENCHMARK_DICT
    FACTSET_AVAILABLE = True
except ImportError:
    FACTSET_AVAILABLE = False
    FactSetCollector = None
    fetch_benchmark = None
    BENCHMARK_DICT = {}

__all__ = [
    'YahooCollector', 
    'FactSetCollector', 
    'fetch_benchmark',
    'BENCHMARK_DICT',
    'FACTSET_AVAILABLE'
]