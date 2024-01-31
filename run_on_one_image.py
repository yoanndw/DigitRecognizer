import numpy as np
import os
import os.path
import sys
from dataset import Dataset, freeman_from_np_2d, load_image_into_2d
from knn import compute_class_with_knn, compute_knn
from naive_bayes import compute_posterior
from cross_validation import cross_val_k

def main():
    args = sys.argv
    if len(args) != 2:
        print(f"ERROR: Not enough or too many arguments\nUsage: {args[0]} <image>")
        print("Exiting.")
        sys.exit(1)

    ## ds = Dataset("../AFAC/")
    ds = Dataset("../projet_ml/ImageMl")

    try:
        img = load_image_into_2d(args[1])
    except FileNotFoundError as e:
        print("ERROR:", e)
        print("Exiting.")
        sys.exit(1)

    freeman = freeman_from_np_2d(img)
    print(freeman)

    print("--------NAIVE BAYES--------")
    print("Digit\tProbability")
    for i in range(0, 10):
        posterior = compute_posterior(ds, freeman, i)
        print(i, posterior, sep="\t")

    print("--------KNN--------")
    # k_nearest = compute_knn(ds, 60, freeman)
    # print(k_nearest)
    nearest_neighbor = compute_class_with_knn(ds, 3, freeman)
    print("Result:", nearest_neighbor)

    print("--------CROSS_VALIDATION--------")
    cv =cross_val_k(ds,freeman, 5)
    print("Result:", cv)
if __name__ == "__main__":
    main()