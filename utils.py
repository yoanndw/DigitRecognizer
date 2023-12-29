# Taken from https://www.kaggle.com/code/mburger/freeman-chain-code-script/
# This code is based on http://www.cs.unca.edu/~reiser/imaging/chaincode.html

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from matplotlib import pyplot as plt
from itertools import chain

from dataset import _np_1d_to_2d

INSERTION_COST = 1
DELETION_COST = 1
EDITION_COST = 2


def freeman_from_np_2d(image):
    """Returns the Freeman code from a 1D dataframe.
    
    Params:
        dataframe: 1D dataframe filled with integers in [0, 255]

    Returns:
        list of int
    """
    # Any results you write to the current directory are saved as output.
    # print("new shape", image.shape)
    # plt.imshow(image, cmap='Greys')

    # plt.imshow(image, cmap='Greys')
    ## Discover the first point 
    for i, row in enumerate(image):
        for j, value in enumerate(row):
            if value == 255:
                start_point = (i, j)
                # print(start_point, value)
                break
        else:
            continue
        break
    image[3:6, 19:22]
    directions = [ 0,  1,  2,
                7,      3,
                6,  5,  4]
    dir2idx = dict(zip(directions, range(len(directions))))

    change_j =   [-1,  0,  1, # x or columns
                -1,      1,
                -1,  0,  1]

    change_i =   [-1, -1, -1, # y or rows
                0,      0,
                1,  1,  1]

    border = []
    chain = []
    curr_point = start_point
    for direction in directions:
        idx = dir2idx[direction]
        new_point = (start_point[0]+change_i[idx], start_point[1]+change_j[idx])
        if image[new_point] != 0: # if is ROI
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
            break

    count = 0
    while curr_point != start_point:
        #figure direction to start search
        b_direction = (direction + 5) % 8 
        dirs_1 = range(b_direction, 8)
        dirs_2 = range(0, b_direction)
        dirs = []
        dirs.extend(dirs_1)
        dirs.extend(dirs_2)
        for direction in dirs:
            idx = dir2idx[direction]
            new_point = (curr_point[0]+change_i[idx], curr_point[1]+change_j[idx])
            if image[new_point] != 0: # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        if count == 1000: break
        count += 1
    # print("count =", count)
    # plt.imshow(image, cmap='Greys')
    # plt.plot([i[1] for i in border], [i[0] for i in border])

    return chain
    

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


def main():
    # train = pd.read_csv("train.csv")

    # print(freeman_from_dataframe(train))

    d = levenshtein("chien", "niche")
    print(d)


if __name__ == "__main__":
    main()