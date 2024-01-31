from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from dataset import Dataset, IMAGE_SIZE
from model_base import ModelBase
from utils import train_test_split

def décision_tree(datasetpath):
    data = pd.read_csv(datasetpath)

    X = data.drop('label', axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()


    model.fit(X_train, y_train)


    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Test accuracy: {accuracy}')

class DT(ModelBase):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.model = DecisionTreeClassifier()

    def train(self, train_set: List[int]):
        self.train_set = train_set

        train_ds = self.dataset.extract_set(train_set).flattened()
        self.model.fit(train_ds.data, train_ds.target)

    def predict_on_dataset(self, dataset: Dataset) -> List[int]:
        return self.model.predict([img.reshape((1, IMAGE_SIZE * IMAGE_SIZE)) for img in dataset.data])

    def predict(self, image):
        return self.model.predict(image.reshape((1, IMAGE_SIZE * IMAGE_SIZE)))[0]

"""
#  to save the trained model
import joblib
joblib.dump(model, 'digit_recognition_model_dt.joblib')
"""

def main():
    ds = Dataset("ImageMl")
    m = DT(ds)
    train, test = train_test_split(ds, 0.3)
    m.train(train)

    print(m.predict(ds.data[test[0]]))

if __name__ == "__main__":
    main()