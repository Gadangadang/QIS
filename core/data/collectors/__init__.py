"""
Data collectors module.
"""

# Only import YahooCollector if yfinance is available
try:
    from .yahoo_collector import YahooCollector
    YAHOO_AVAILABLE = True
except ImportError:
    YAHOO_AVAILABLE = False
    YahooCollector = None

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
    'YAHOO_AVAILABLE',
    'FactSetCollector',
    'fetch_benchmark',
    'BENCHMARK_DICT',
    'FACTSET_AVAILABLE'
]