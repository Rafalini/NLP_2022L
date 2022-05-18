import pandas as pd

from models.abstract_classifier import TextEntityClassifier


class OurTrainer:
    def __init__(self, model: TextEntityClassifier):
        self.model = model

    def train(self, raw_data: pd.DataFrame):
        data = self.model.prepare_training_data(raw_data)
        self.model.train(data)
