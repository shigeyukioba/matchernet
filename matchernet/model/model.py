import os
import random
import numpy as np
import pandas as pd
from collections import OrderedDict

import keras
from keras import backend as K
from keras import (Model, Input, optimizers, losses)
from keras.layers import (
    Activation, Dense, Dropout, AlphaDropout, Conv1D, Conv2D, MaxPool1D,
    MaxPool2D, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalAveragePooling2D,
    Add, Maximum, Concatenate, BatchNormalization, Embedding, Reshape, Lambda)

from matchernet.model import feature_extraction


def merge_layer(layers, output_size):
    x = Concatenate()(layers)
    x = Dense(output_size, activation='relu')(x)
    return x


class ModelConstruction(object):
    """model
    """

    def __init__(self,
                 inputs_df,
                 target_column=None,
                 share_modal_layers=False,
                 **params):
        self.inputs_df = inputs_df

        self.target_column = self._validate_target_column(target_column)
        self.share_modal_layers = share_modal_layers
        self.inputs_dict = OrderedDict()
        self.feature_extraction_layers = []
        self.inputs_layers = []

        self.model_type = None

        self.img_shape = params["img_shape"] if "img_shape" in params else (
            128, 128, 3)

        self.num_mlp = params[
            "num_mlp"] if "num_mlp" in params else feature_extraction.numerical_mlp
        self.img_feature_extraction = params[
            "img_feature_extraction"] if "img_feature_extraction" in params else feature_extraction.image_xception
        self.merge_layer = params[
            "merge_layer"] if "merge_layer" in params else merge_layer

        self.feature_output_size = params[
            "feature_output_size"] if "feature_output_size" in params else 256

        self.modal_layer_dict = params[
            "modal_layer_dict"] if "modal_layer_dict" in params else None

    def __call__(self):
        return self.define_feature_extraction_layer()

    def define_feature_extraction_layer(self):
        num_inputs = []
        for i, c in enumerate(self.inputs_df.columns):
            if self.modal_layer_dict is not None:
                if c in self.modal_layer_dict.keys():
                    self._add_unique_layer(c,
                                           self.modal_layer_dict[c]["format"])
                    continue
            if c.endswith("num"):
                num_inputs.append(c)
            elif c.endswith("img"):
                self._add_img_layer(c, "img")
            elif c.endswith("img_path"):
                self._add_img_layer(c, "img_path")
            else:
                raise NotImplementedError
        if i == len(self.inputs_df.columns) - 1 and len(num_inputs) > 0:
            self._add_num_layer(num_inputs, "num")

        if len(self.feature_extraction_layers) == 0:
            raise KeyError("No columns for feature extraction")
        elif len(self.feature_extraction_layers) == 1:
            return self.feature_extraction_layers[0]
        else:
            return self.merge_layer(
                self.feature_extraction_layers,
                output_size=self.feature_output_size)

    def _add_layer(self, inputs, x, c, format):
        self.feature_extraction_layers.append(x)
        self.inputs_layers.append(inputs)
        self.inputs_dict[inputs.name] = {"format": format, "column": c}

    def _add_img_layer(self, c, format):
        inputs = Input(shape=self.img_shape)
        x = self.img_feature_extraction(
            inputs, output_size=self.feature_output_size)
        self._add_layer(inputs, x, c, format)

    def _add_num_layer(self, num_inputs, format):
        inputs = Input(shape=(len(num_inputs),))
        x = self.num_mlp(inputs, output_size=self.feature_output_size)
        self._add_layer(inputs, x, num_inputs, format)

    def _add_unique_layer(self, c, format):
        self.inputs_layers.append(self.modal_layer_dict[c]["inputs"])
        self.feature_extraction_layers.append(
            self.modal_layer_dict[c]["feature_extraction"])
        self.inputs_dict[self.modal_layer_dict[c]["inputs"].name] = {
            "format": format,
            "column": c
        }

    def _validate_target_column(self, target_column):
        if target_column is None:
            return None
        if target_column in self.inputs_df.columns:
            return target_column
        else:
            raise KeyError("{0} not in inputs_df".format(target_column))


class Classifier(ModelConstruction):
    """multi-class classifier
    """

    def __init__(self,
                 inputs_df,
                 target_column=None,
                 num_classes=None,
                 share_modal_layers=False,
                 **params):
        super().__init__(inputs_df, target_column, share_modal_layers,
                         **params)
        self._num_classes = num_classes
        self.num_classes = self._get_target_class_num()

        self.model_type = "multiclassifier"

        self.classifier = params[
            "classifier"] if "classifier" in params else self._classifier

    def __call__(self):
        x = self.define_feature_extraction_layer()
        x = self.classifier(x, num_classes=self.num_classes)
        model = ModelConstruction(self.inputs_layers, x)
        return model

    def _get_target_class_num(self):
        if self.target_column is None:
            if self._num_classes is None:
                raise ValueError(
                    "Either num_classes or target_column should be specified.")
            else:
                return self._num_classes
        else:
            if self.target_column not in self.inputs_df.columns:
                raise KeyError("No '{0}' column in input dataframe".format(
                    self.target_column))

            def _to_categorical():
                return keras.utils.to_categorical(
                    self.inputs_df[self.target_column].values)

            _targets = _to_categorical()
            if self._num_classes is not None:
                if self._num_classes != len(_targets[0]):
                    raise ValueError(
                        "num_classes and classes in target column don't match")
                else:
                    return self._num_classes
            else:
                return len(_targets[0])

    def _classifier(self, x, num_classes):
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)
        return x


class Regressor(ModelConstruction):
    """multimodal regressor
    """

    def __init__(self,
                 inputs_df,
                 target_column=None,
                 share_modal_layers=False,
                 **params):
        super().__init__(inputs_df, target_column, share_modal_layers,
                         **params)

        self._validate_target()

        self.model_type = "regressor"

        self.activation = params[
            "activation"] if "activation" in params else "linear"
        self.regressor = params[
            "regressor"] if "regressor" in params else self._regressor

    def __call__(self):
        x = self.define_feature_extraction_layer()
        x = self.regressor(x, activation=self.activation)
        model = ModelConstruction(self.inputs_layers, x)
        return model

    def _validate_target(self):
        if self.target_column is not None:
            for i in self.inputs_df[self.target_column]:
                if not isinstance(i, float) and not isinstance(i, int):
                    raise ValueError(
                        "target column should be either int or float")

    def _regressor(self, x, activation):
        x = Dropout(0.5)(x)
        x = Dense(1, activation=activation)(x)
        return x


class BinaryClassifier(ModelConstruction):
    """binary classifier
    """

    def __init__(self,
                 inputs_df,
                 target_column=None,
                 share_modal_layers=False,
                 **params):
        super().__init__(inputs_df, target_column, share_modal_layers,
                         **params)

        self._validate_target()

        self.model_type = "binaryclassifier"

        self.binary_classifier = params[
            "binary_classifier"] if "binary_classifier" in params else self._binary_classifier

    def __call__(self):
        x = self.define_feature_extraction_layer()
        x = self.binary_classifier(x)
        model = ModelConstruction(self.inputs_layers, x)
        return model

    def _validate_target(self):
        if self.target_column is not None:

            def _to_categorical():
                return keras.utils.to_categorical(
                    self.inputs_df[self.target_column].values)

            _targets = _to_categorical()
            if len(_targets[0]) > 2:
                raise ValueError(
                    "target column should have only 2 values; e.g. 0 and 1")

    def _binary_classifier(self, x):
        x = Dropout(0.5)(x)
        x = Dense(1, activation="softmax")(x)
        return x
