import numpy as np
import keras

from .image_generator import image_generator


class MultiModalIterator(object):
    def __init__(self,
                 data_df,
                 target_column,
                 model_type="multiclass",
                 train=True,
                 batch_size=8,
                 shuffle=True,
                 imagegen=image_generator(),
                 **params):

        self.data_df = data_df
        self.train = train
        self.model_type = model_type
        self.target_column = self._validate_target_column(target_column)
        self.targets = self._get_target()
        self.sample_num = len(self.data_df)
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.imagegen = imagegen

        self.image_shape = params["image_shape"] \
            if "image_shape" in params else (299, 299, 3)

    def __call__(self):
        while True:
            indexes = self._get_indexes()
            iteration_num = int(len(indexes) // self.batch_size)

            for i in range(iteration_num):
                batch_ids = indexes[i * self.batch_size:(i + 1) * self.batch_size]
                inputs, targets = self._data_generation(batch_ids)

                yield inputs, targets

    def _get_target(self):
        """Select a label according to model_type
        """
        if self.model_type == "multiclass":
            return self._to_categorical()
        else:
            raise NotImplementedError

    def _data_generation(self, batch_ids):
        """Gather data for each modality
        """
        inputs, num_inputs = [], []
        for i, col in enumerate(self.data_df.columns):
            if col.endswith("image") or col.endswith("image_path"):
                x = self._load_image(self.data_df[col].values[batch_ids])
                inputs.append(x)

        targets = self.targets[batch_ids]

        return inputs, targets

    def _get_indexes(self):
        """Create and shuffle data index

        Returns:
            indexes(numpy.array): (sample_num), Data index
        """
        indexes = np.arange(self.sample_num)
        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def _to_categorical(self):
        return keras.utils.to_categorical(self.data_df[self.target_column].values)

    def _validate_target_column(self, target_column):
        """Check if specified target_column exists
        """
        if target_column not in self.data_df.columns:
            raise KeyError("{0} not in DataFrame.".format(target_column))
        else:
            return target_column

    def _load_image(self, paths):
        """Loading image data from a path

        The single image tensor is extended at random according to the augmentation
        and parameters specified by random_transform of ImageDataGenerator, and normalized
        by standardize.
        Use for both train and test, but perform data augmentation only during train.

        Args:
            paths(list): Image data path

        Returns:
            x(numpy.array): Preprocessed image data
        """
        x = np.zeros((len(paths), *self.image_shape))
        for i, path in enumerate(paths):
            if path is not np.nan:
                x[i] = np.load(path)["img"]
        x = x.astype("float32")
        if self.train:
            for i in range(self.batch_size):
                x[i] = self.imagegen.random_transform(x[i])
                x[i] = self.imagegen.standardize(x[i])
        x /= 255

        return x
