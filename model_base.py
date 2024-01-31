from typing import List

from dataset import Dataset

class ModelBase:
    def predict_on_dataset(self, dataset: Dataset) -> List[int]:
        predictions = []
        for i in range(len(dataset.data)):
            predictions.append(self.predict(dataset.freeman[i]))

        return predictions
    
    def accuracy(self, dataset: Dataset) -> float:
        preds = self.predict_on_dataset(dataset)
        return len([c for (p, c) in zip(preds, dataset.target) if p == c]) / len(dataset.target)