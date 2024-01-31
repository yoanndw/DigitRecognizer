import os
import os.path
from typing import List

import cv2
from keras.datasets import mnist
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold

IMAGE_SIZE = 28

def _open_image(path):
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.NEAREST)
    return image

def _image_to_np_2d(image):
    return _np_1d_to_2d(_image_to_np_1d(image))

def _np_1d_to_2d(array):
    return array.reshape((IMAGE_SIZE, IMAGE_SIZE))

def _image_to_np_1d(image):
    return np.float32(np.array(image.getdata()))

def load_image_into_2d(path):
    image = _open_image(path)
    arr = _image_to_np_2d(image)
    ret, image = cv2.threshold(arr, 127, 255, 0)
    return image


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
        try:
            if image[new_point] != 0: # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        except IndexError:
            pass

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
            try:
                if image[new_point] != 0: # if is ROI
                    border.append(new_point)
                    chain.append(direction)
                    curr_point = new_point
                    break
            except IndexError:
                pass
        if count == 1000: break
        count += 1
    # print("count =", count)
    # plt.imshow(image, cmap='Greys')
    # plt.plot([i[1] for i in border], [i[0] for i in border])

    return chain


class Dataset:
    def __init__(self, directory_path=None):
        self.data = []
        self.freeman = []
        self.target = []

        if directory_path is not None:
            for filename in os.listdir(directory_path):
                path = os.path.join(directory_path, filename)
                image = load_image_into_2d(path)
                freeman = freeman_from_np_2d(image)
                target = int(filename[0])
                self.data.append(image)
                self.freeman.append(freeman)
                self.target.append(target)

    def load_mnist(self):
        (train_data, train_target), (test_data, test_target) = mnist.load_data()
        images = []
        images.extend(train_data)
        images.extend(test_data)
        images = [img.reshape((IMAGE_SIZE, IMAGE_SIZE)) for img in images]
        images = [cv2.threshold(img, 127, 255, 0)[1] for img in images]

        targets = []
        targets.extend(train_target)
        targets.extend(test_target)
        
        self.data.extend(images)
        self.target.extend(targets)
        self.freeman.extend([freeman_from_np_2d(img) for img in images])

    def tuples(self):
        """Return a list of tuples with `(ndarray of the image, freeman code, target)`"""
        
        return [(self.data[i], self.freeman[i], self.target[i]) for i in range(len(self.data))]

    def extract_set(self, set: List[int]):
        new_ds = Dataset()
        for i in set:
            new_ds.data.append(self.data[i])
            new_ds.target.append(self.target[i])
            new_ds.freeman.append(self.freeman[i])
        return new_ds

    def cross_val_split(self, n_splits: int):
        kf = KFold(n_splits=n_splits)
        return kf.split(self.data)


def main():
    np.set_printoptions(threshold=np.inf)
    ds = Dataset("../projet_ml/ImageMl")
    ds.load_mnist()
    for i in range(len(ds.data)):
        print(ds.target[i], len(ds.freeman[i]), ds.freeman[i])
if __name__ == "__main__":
    main()