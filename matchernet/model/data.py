import numpy as np


def data_load(img_path):
    x_img = np.zeros((len(img_path), 128, 128, 3))
    for i, p in enumerate(img_path):
        if p is not np.nan:
            x_img[i] = np.load(p)["img"]
    return x_img


def normalize(x_img):
    x_img = x_img.astype('float32')
    x_img /= 255
    return x_img
