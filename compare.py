from typing import List

from dataset import Dataset

class MetricsForClass:
    def __init__(self, cls, tn, tp, fn, fp):
        self.cls = cls
        self.tn = tn
        self.tp = tp
        self.fn = fn
        self.fp = fp

        self.precision = tp / (fp + tp)
        self.recall = tp / (fn + tp)
        self.specificity = tn / (tn + fp)


class Metrics:
    def __init__(self, accurate_predictions, matrix_sum, metrics_by_class):
        self.accurate_predictions = accurate_predictions
        self.matrix_sum = matrix_sum
        self.metrics_by_class = metrics_by_class

        self.matrix_size = len(metrics_by_class)
        self.accuracy = accurate_predictions / matrix_sum


def compute_confusion_matrix(train_ds: Dataset, test_ds: Dataset, model, **additional_params):
    matrix = []
    for i in range(10):
        matrix[i] = []
        for j in range(10):
            matrix[i][j] = 0

    for (d, f, t) in test_ds.tuples():
        predicted = model(train_ds, freeman=f, **additional_params)
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


def main():
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


if __name__ == "__main__":
    main()