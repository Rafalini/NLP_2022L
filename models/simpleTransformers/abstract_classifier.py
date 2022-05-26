from abc import ABC, abstractmethod
import pandas as pd


class TextEntityClassifier(ABC):
    @abstractmethod
    def prepare_training_data(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def predict(self, inputs):
        pass
