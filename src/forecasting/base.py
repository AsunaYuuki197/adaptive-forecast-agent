from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def train_and_predict(self, data: pd.DataFrame):
        pass
