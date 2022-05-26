import pandas as pd
from simpletransformers.classification import ClassificationModel

from models.abstract_classifier import TextEntityClassifier


class BertWrapper(TextEntityClassifier):
    def __init__(self, model: ClassificationModel):
        self.model: ClassificationModel = model

    def train(self, data):
        self.model.train_model(
            train_df=data,
            show_running_loss=False
        )

    def predict(self, inputs):
        predictions, raw_outputs = self.model.predict(inputs)
        return predictions

    def prepare_training_data(self, raw_data: pd.DataFrame):
        data = raw_data.copy()
        # print(data.head)
        # 'text' and 'labels' columns already present, no need to do anything
        data = data.astype({'text': str, 'entityTag': int})
        return data[['text', 'entityTag']]
