from typing import List

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, cross_val_score


from dataset import Dataset
from knn import Knn

def perform_cross_validation(X, y, model, cv=5):
    """
    Perform cross-validation on a given machine learning model.
    Parameters:
    - X: Features (feature matrix)
    - y: Target variable
    - model: Machine learning model
    - cv: Number of folds for cross-validation (default is 5)

    Returns:
    - Mean accuracy across folds
    - Standard deviation of accuracy across folds
    """

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=cv)

    # Print the cross-validation scores
    print("Cross-Validation Scores:", scores)

    # Print the mean and standard deviation of the scores
    mean_accuracy = scores.mean()
    std_dev = scores.std()
    print(f"Mean Accuracy: {mean_accuracy:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")

    return mean_accuracy, std_dev

def cross_val_k(dataset: Dataset, ks: List[int], n_splits: int):
    accuracies = {}
    for k in ks:
        mean_accuracy = 0.
        for train, test in dataset.cross_val_split(n_splits):
            clf = Knn(dataset, k)
            clf.train(train)
            mean_accuracy += clf.accuracy(dataset.extract_set(test))
        mean_accuracy /= n_splits
        accuracies[k] = mean_accuracy

    sorted_accuracies = sorted(accuracies.items(), key=lambda e: e[1], reverse=True)
    # print(sorted_accuracies)
    return sorted_accuracies

def main():
    ds = Dataset("ImageMl/")
    # ds.load_sklearn()
    results = cross_val_k(ds, [3, 4, 5, 8, 10], 8)
    print("Results", results)

if __name__ == "__main__":
    main()