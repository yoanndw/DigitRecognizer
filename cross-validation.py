import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn import svm

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

