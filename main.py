from dataset import load_image_into_2d
from utils import freeman_from_np_2d

import cv2
import numpy as np
from PIL import Image

def main():
    np.set_printoptions(threshold=np.inf)
    array = load_image_into_2d("../AFAC/3_0.png")
    print(array)
    freeman = freeman_from_np_2d(array)
    print(freeman)


if __name__ == "__main__":
    main()