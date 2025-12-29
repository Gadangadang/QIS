"""
Base Data Processor Interface.
"""

from abc import ABC, abstractmethod
import pandas as pd

class BaseProcessor(ABC):
    """
    Abstract base class for data processors.
    """

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input DataFrame.

        Args:
            data: Raw DataFrame.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        pass
