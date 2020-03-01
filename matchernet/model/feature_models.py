import os
import random
import numpy as np
import pandas as pd

import keras
from keras import backend as K
from keras import (Model, Input, optimizers, losses, callbacks)
from keras.layers import (Activation, Dropout, AlphaDropout, Conv1D, Conv2D,
                          Reshape, Lambda, GlobalMaxPooling1D, MaxPool2D,
                          GlobalAveragePooling2D, Dense, MaxPool1D,
                          GlobalMaxPooling2D, BatchNormalization, Embedding,
                          Concatenate, Maximum, Add)


def numerical_mlp(inputs, output_size):
    x = Dense(512, activation="relu")(inputs)
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(output_size)(x)
    return x


def image_xception(inputs, output_size):
    cnn = keras.applications.Xception(
        input_tensor=inputs, include_top=False, weights='imagenet')
    x = cnn.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(output_size, activation='relu')(x)
    return x


def image_inception_v3(inputs, output_size):
    cnn = keras.applications.InceptionV3(
        input_tensor=inputs, include_top=False, weights='imagenet')
    x = cnn.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(output_size, activation='relu')(x)
    return x


def image_inception_resnet_v2(inputs, output_size):
    cnn = keras.applications.InceptionResNetV2(
        input_tensor=inputs, include_top=False, weights='imagenet')
    x = cnn.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(output_size, activation='relu')(x)
    return x
