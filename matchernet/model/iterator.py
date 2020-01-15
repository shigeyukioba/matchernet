import os
import random
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import cv2
import re
import unicodedata
from collections import OrderedDict

import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator


# random erasing for image
def get_random_eraser(p=0.5,
                      s_l=0.02,
                      s_h=0.4,
                      r_1=0.3,
                      r_2=1 / 0.3,
                      v_l=0,
                      v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


def image_gen(rotation_range=180,
              width_shift_range=0.2,
              height_shift_range=0.2,
              shear_range=10,
              zoom_range=0.3,
              horizontal_flip=True,
              vertical_flip=True,
              channel_shift_range=5.,
              brightness_range=[0.3, 1.0],
              preprocessing_function=get_random_eraser(v_l=0, v_h=255)):
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        channel_shift_range=channel_shift_range,
        brightness_range=brightness_range,
        preprocessing_function=preprocessing_function)


class Iterator(object):
    def __init__(self,
                 data_df,
                 target_column,
                 train=True,
                 model_type="multiclassifier",
                 batch_size=8,
                 shuffle=True,
                 imagegen=image_gen(),
                 **params):

        self.data_df = data_df
        self.train = train
        self.target_column = self._validate_target_column(target_column)
        self.model_type = model_type
        self.targets = self._get_target()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_num = len(self.data_df)

        self.imagegen = imagegen
        self.img_shape = params["img_shape"] if "img_shape" in params else (
            128, 128, 3)

    def __call__(self):
        while True:
            indexes = self._get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size:(i + 1) *
                                                        self.batch_size]
                inputs, targets = self._data_generation(batch_ids)

                yield inputs, targets

    def _validate_target_column(self, target_column):
        if target_column not in self.data_df.columns:
            raise KeyError("{0} not in data_df".format(target_column))
        else:
            return target_column

    def _get_target(self):
        if self.model_type == "multiclassifier":
            return self._to_categorical()
        elif self.model_type == "regressor":
            return self._to_regression()
        elif self.model_type == "binaryclassifier":
            return self._to_binary()

    def _to_categorical(self):
        # to categorical
        return keras.utils.to_categorical(
            self.data_df[self.target_column].values)

    def _to_regression(self):
        return np.array([[x] for x in self.data_df[self.target_column].values])

    def _to_binary(self):
        return np.array([[x] for x in self.data_df[self.target_column].values])

    def _get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            # get shuffled indexes
            np.random.shuffle(indexes)

        return indexes

    def _load_img(self, x):
        # load image
        _x = np.zeros((len(x), *self.img_shape))
        for i, p in enumerate(x):
            if p is not None:
                # load only if exists
                _x[i] = np.load(p)["img"]
        _x = _x.astype('float32')
        if self.train:
            # data augmentations
            for i in range(self.batch_size):
                _x[i] = self.imagegen.random_transform(_x[i])
                _x[i] = self.imagegen.standardize(_x[i])
        _x /= 255
        return _x

    def _data_generation(self, batch_ids):
        inputs = []
        num_inputs = []
        for i, c in enumerate(self.data_df.columns):
            if c.endswith("num"):
                num_inputs.append(self.data_df[c].values[batch_ids])
            elif c.endswith("img") or c.endswith("img_path"):
                x = self._load_img(self.data_df[c].values[batch_ids])
                inputs.append(x)
            else:
                raise NotImplementedError
        if len(num_inputs) > 0:
            inputs.append(np.array(num_inputs).T)

        targets = self.targets[batch_ids]
        return inputs, targets


class ModelIterator(Iterator):
    def __init__(self,
                 model,
                 data_df,
                 target_column,
                 train=True,
                 batch_size=8,
                 shuffle=True,
                 imagegen=image_gen(),
                 **params):
        super().__init__(
            data_df=data_df,
            target_column=target_column,
            train=train,
            batch_size=batch_size,
            shuffle=shuffle,
            imagegen=imagegen,
            **params)
        self.model = model
        self.target_column = self._get_target_column(target_column)
        self.model_type = self.model.model_type
        self.targets = self._get_target()

    def _get_target_column(self, target_column):
        _target_column = self._validate_target_column(target_column)
        if _target_column != self.model.target_column:
            raise ValueError(
                "target_column {0} is different from model target_column {1}".
                    format(_target_column, self.model.target_column))
        else:
            return _target_column

    def _data_generation(self, batch_ids):
        inputs = []
        num_inputs = []

        def _format_branch(format, column):
            if format == "num":
                x = np.array(
                    [[x] for x in self.data_df[column].values[batch_ids]])
                inputs.append(x)
            elif format == "img" or format == "img_path":
                x = self._load_img(self.data_df[column].values[batch_ids])
                inputs.append(x)
            else:
                raise NotImplementedError

        for k, v in self.model.inputs_dict.items():
            if self.model.modal_layer_dict is not None:
                if isinstance(v["column"], str):
                    if v["column"] in self.model.modal_layer_dict.keys(
                    ):
                        _format_branch(v["format"], v["column"])
                        continue
            if isinstance(v["column"], list):
                if v["format"] == "num":
                    for c in v["column"]:
                        num_inputs.append(self.data_df[c].values[batch_ids])
            else:
                _format_branch(v["format"], v["column"])
        if len(num_inputs) > 0:
            inputs.append(np.array(num_inputs).T)

        targets = self.targets[batch_ids]
        return inputs, targets
