from typing import List
from dataset import Dataset
from utils import levenshtein

def compute_knn(ds: Dataset, k: int, freeman: List[int]):
    ds_with_distance = [(f, t, levenshtein(f, freeman)) for (d, f, t) in ds.tuples()]
    sorted_distances = sorted(ds_with_distance, key=lambda t: t[2])
    print("Digit\tDistance\tFreeman")
    for (f, t, dist) in sorted_distances:
        print(t, dist, f, sep="\t")
    k_nearest = sorted_distances[:k]
    return k_nearest

def compute_class_with_knn(ds: Dataset, k: int, freeman: List[int]):
    k_sorted = compute_knn(ds, k, freeman)

    max_class_count = 0
    most_frequent_class = None
    for i in range(0, 10):
        same_class_elems = [(f, t, dist) for (f, t, dist) in k_sorted if t == i]
        count = len(same_class_elems)
        if count > max_class_count:
            max_class_count = count
            most_frequent_class = i

    return most_frequent_class
