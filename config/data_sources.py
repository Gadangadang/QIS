"""
Data source configuration for easy switching between Yahoo and FactSet.
"""

import os

# Check if running in FactSet environment
RUNNING_IN_FACTSET = os.getenv('FACTSET_ENVIRONMENT', 'false').lower() == 'true'

# Default data source
DEFAULT_DATA_SOURCE = 'factset' if RUNNING_IN_FACTSET else 'yahoo'

# Data source settings
DATA_SOURCE_CONFIG = {
    'use_factset': DEFAULT_DATA_SOURCE == 'factset',
    'cache_enabled': True,
    'cache_dir': 'Dataset'
}

def get_data_source():
    """Get configured data source name."""
    return DEFAULT_DATA_SOURCE

def should_use_factset():
    """Check if FactSet should be used."""
    return DATA_SOURCE_CONFIG['use_factset']