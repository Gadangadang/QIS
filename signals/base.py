from abc import ABC, abstractmethod


class SignalModel(ABC):
    @abstractmethod
    def generate(self, df):
        raise NotImplementedError
