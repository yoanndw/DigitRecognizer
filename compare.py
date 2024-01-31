from typing import List
import random

from dataset import Dataset
from knn import Knn
from model_base import ModelBase
from naive_bayes import NaiveBayes

class MetricsForClass:
    def __init__(self, cls, tn, tp, fn, fp):
        self.cls = cls
        self.tn = tn
        self.tp = tp
        self.fn = fn
        self.fp = fp

        if fp + tp == 0:
            self.precision = None
        else:
            self.precision = tp / (fp + tp)
        
        if fn+tp == 0:
            self.recall = None
        else:
            self.recall = tp / (fn + tp)
        
        if tn+fp == 0:
            self.specificity = None
        else:
            self.specificity = tn / (tn + fp)


class Metrics:
    def __init__(self, accurate_predictions, matrix_sum, metrics_by_class):
        self.accurate_predictions = accurate_predictions
        self.matrix_sum = matrix_sum
        self.metrics_by_class = metrics_by_class

        self.matrix_size = len(metrics_by_class)
        self.accuracy = accurate_predictions / matrix_sum

        # Do not consider None values in mean computation
        metrics_by_class = [mbc for mbc in metrics_by_class if mbc.precision is not None and mbc.specificity is not None and mbc.recall is not None]

        total_precision = 0
        total_recall = 0
        total_specificity = 0
        for mbc in metrics_by_class:
            total_precision += mbc.precision
            total_recall += mbc.recall
            total_specificity += mbc.specificity

        self.precision = total_precision / len(metrics_by_class)
        self.recall = total_recall / len(metrics_by_class)
        self.specificity = total_specificity / len(metrics_by_class)


def compute_confusion_matrix(train_set: List[int], test_ds: Dataset, model: ModelBase):
    matrix = []
    for i in range(10):
        matrix.append([])
        for j in range(10):
            matrix[i].append(0)

    model.train(train_set)
    for (d, f, t) in test_ds.tuples():
        predicted = model.predict(f)
        matrix[t][predicted] += 1

    return matrix


def compute_metrics(confusion_matrix: List[List[int]]):
    # Count accurate predictions
    accurate_predictions = 0
    matrix_sum = 0
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[0])):
            matrix_sum += confusion_matrix[i][j]
            if i == j:
                accurate_predictions += confusion_matrix[i][i]
    
    # TP, TN, FP, FN
    results_by_class = []
    for cls in range(len(confusion_matrix)):
        tn = 0
        tp = 0
        fn = 0
        fp = 0
        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[0])):
                m = confusion_matrix[i][j]

                if i == cls:
                    if j == cls:
                        tp += m
                    else:
                        fn += m
                else:
                    if j == cls:
                        fp += m
                    else:
                        tn += m
        
        results_by_class.append(MetricsForClass(cls, tn, tp, fn, fp))

    return Metrics(accurate_predictions, matrix_sum, results_by_class)

def train_test_split(dataset: Dataset, test_size: float):
    train_indices = []
    test_indices = []
    for i in range(len(dataset.data)):
        if random.random() <= test_size:
            test_indices.append(i)
        else:
            train_indices.append(i)

    return train_indices, test_indices

def main_compute_from_matrix():
    matrix = [
        [60, 30],
        [80, 20]
    ]

    metrics = compute_metrics(matrix)
    print("accuracy", metrics.accuracy)
    for cls, m in enumerate(metrics.metrics_by_class):
        print("------ Class:", cls, "--------")
        print("precision", m.precision)
        print("recall", m.recall)
        print("specificity", m.specificity)


def main():
    ds = Dataset("ImageMl")

    m = Knn(ds, 3)

    train, test = train_test_split(ds, 0.3)
    print(train, test)
    print(len(train), len(test))

    m.train(train)
    matrix = compute_confusion_matrix(train, ds.extract_set(test), m)
    print(matrix)
    metrics: Metrics = compute_metrics(matrix)
    for cls, m in enumerate(metrics.metrics_by_class):
        print("------ Class:", cls, "--------")
        print("precision", m.precision)
        print("recall", m.recall)
        print("specificity", m.specificity)

    print("------ MEAN --------")
    print("accuracy", metrics.accuracy)
    print("precision", metrics.precision)
    print("recall", metrics.recall)
    print("specificity", metrics.specificity)

    

if __name__ == "__main__":
    main()