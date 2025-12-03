"""
Futures trading module.

Handles futures-specific operations:
- Contract rollover management
- Expiration date calculation
- Continuous series generation
- Rollover cost tracking
"""

from core.futures.rollover_handler import FuturesRolloverHandler, EXPIRATION_RULES

__all__ = [
    'FuturesRolloverHandler',
    'EXPIRATION_RULES',
]
