# Taken from https://www.kaggle.com/code/mburger/freeman-chain-code-script/
# This code is based on http://www.cs.unca.edu/~reiser/imaging/chaincode.html

import random

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from matplotlib import pyplot as plt
from itertools import chain

from dataset import _np_1d_to_2d

INSERTION_COST = 1
DELETION_COST = 1
EDITION_COST = 2

def _cost(source, dest):
    if dest == '': # deletion
        return DELETION_COST
    if source == '': # insertion
        return INSERTION_COST
    if source == dest:
        return 0
    
    return EDITION_COST


def levenshtein(s1, s2):
    len_s1 = len(s1)
    len_s2 = len(s2)

    n_rows = len_s1 + 1
    n_cols = len_s2 + 1

    distance = []
    for i in range(n_rows):
        distance.append([0] * n_cols)
        distance[i][0] = i

    for j in range(len_s2 + 1):
        distance[0][j] = j
    
    for i in range(1, n_rows):
        for j in range(1, n_cols):
            c1 = s1[i - 1]
            c2 = s2[j - 1]
            d1 = distance[i - 1][j - 1] + _cost(c1, c2) 
            d2 = distance[i - 1][j] + _cost(c1, '') 
            d3 = distance[i][j - 1] + _cost('', c2) 
            distance[i][j] = min(d1, d2, d3)

    return distance[len_s1][len_s2]


def train_test_split(dataset, test_size: float):
    train_indices = []
    test_indices = []
    for i in range(len(dataset.data)):
        if random.random() <= test_size:
            test_indices.append(i)
        else:
            train_indices.append(i)

    return train_indices, test_indices


def main():
    # train = pd.read_csv("train.csv")

    # print(freeman_from_dataframe(train))

    d = levenshtein("chien", "niche")
    print(d)


if __name__ == "__main__":
    main()