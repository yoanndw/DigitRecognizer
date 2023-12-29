import numpy as np
import os
import os.path

from dataset import Dataset, freeman_from_np_2d, load_image_into_2d
from naive_bayes import compute_posterior

def main():
    np.set_printoptions(threshold=np.inf)
    # img = load_image_into_2d("../train/1_0.png")
    # freeman = freeman_from_np_2d(img)
    # print(freeman)

    one_freeman = [3, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 6, 5, 5, 5, 5, 6, 5, 5, 5, 5, 4, 3, 5, 7, 7, 7, 7, 7, 7, 7, 7, 1, 3, 3, 3, 3, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 6, 7, 6, 6, 6, 6, 7, 7, 2, 2, 2, 3, 2, 2, 2, 2, 2]

    ds = Dataset("../AFAC/")
    for i in range(10):
        posterior = compute_posterior(ds, one_freeman, i)
        print(i, posterior)

if __name__ == "__main__":
    main()