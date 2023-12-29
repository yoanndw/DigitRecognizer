import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = 64

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



def main():
    im_frame = Image.open("../AFAC/3_0.png")
    im_frame = im_frame.resize((64, 64), resample=Image.Resampling.NEAREST)
    im_frame.show()

    np_frame = np.array(im_frame.getdata())

    print(np_frame.shape)
    print(np_frame)


if __name__ == "__main__":
    main()