"""
Base Feature Generator Interface.
"""

from abc import ABC, abstractmethod
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BaseFeatureGenerator(ABC):
    """
    Abstract base class for all feature generators.
    Enforces a consistent interface for creating features.
    """

    @abstractmethod
    def generate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate features from the input data.

        Args:
            data: Input DataFrame (e.g., prices, macro data).
            **kwargs: Additional arguments (e.g., benchmark data).

        Returns:
            pd.DataFrame: DataFrame containing the generated features.
                          Index should align with the input data.
        """
        pass
