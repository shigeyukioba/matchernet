from collections import OrderedDict
import keras

from .feature_models import image_feature


class MultiModalModel(object):
    def __init__(self,
                 inputs_df,
                 target_column,
                 feature_output_size=256,
                 **params):
        self.inputs_df = inputs_df
        self.targets = self._get_targets(target_column)
        self.feature_extraction_layers = []
        self.feature_output_size = feature_output_size
        self.inputs_dict = OrderedDict()
        self.inputs_layers = []

        self.image_shape = params["image_shape"] \
            if "image_shape" in params else (299, 299, 3)

        self.image_feature = params["image_feature"] \
            if "image_feature" in params else image_feature

    def __call__(self):
        return self.feature_extraction()

    def feature_extraction(self):
        for i, col in enumerate(self.inputs_df.columns):
            if col.endswith("image"):
                self._add_image_feature_layer(col, "image")
            elif col.endswith("image_path"):
                self._add_image_feature_layer(col, "image_path")
            else:
                continue

        if len(self.feature_extraction_layers) == 0:
            raise KeyError("No feature extraction layer.")
        elif len(self.feature_extraction_layers) == 1:
            return self.feature_extraction_layers[0]
        else:
            return self.merge_layer(
                layers=self.feature_extraction_layers,
                output_size=self.feature_output_size
            )

    def merge_layer(self, layers, output_size):
        x = keras.layers.Concatenate()(layers)
        x = keras.layers.Dense(output_size, activation="relu")(x)

        return x

    def _add_layer(self, inputs, x, col, format):
        self.feature_extraction_layers.append(x)
        self.inputs_layers.append(inputs)
        self.inputs_dict[inputs.name] = {"format": format, "column": col}

    def _add_image_feature_layer(self, c, format):
        inputs = keras.Input(shape=self.image_shape)
        x = self.image_feature(
            inputs=inputs,
            output_size=self.feature_output_size
        )
        self._add_layer(inputs, x, c, format)

    def _validate_target_column(self, target_column):
        if target_column in self.inputs_df.columns:
            return target_column
        else:
            raise KeyError("{0} is not in input DataFrame.".format(target_column))

    def _get_targets(self, target_column):
        target_column = self._validate_target_column(target_column)
        return keras.utils.to_categorical(self.inputs_df[target_column].values)


class MultiModalClassifier(MultiModalModel):
    def __init__(self,
                 inputs_df,
                 num_classes,
                 target_column="target"):
        super().__init__(inputs_df, target_column)
        self.num_classes = num_classes

    def __call__(self):
        x = self.feature_extraction()
        x = self._classifier(x)
        model = keras.Model(self.inputs_layers, x)

        return model

    def _classifier(self, x):
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(self.num_classes, activation="softmax")(x)

        return x
