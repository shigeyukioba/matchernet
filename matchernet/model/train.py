import os
import datetime
import numpy as np
import pandas as pd
from keras import utils
from keras import Model, Input
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import save_model, load_model
from keras.layers import Activation, Dropout, AlphaDropout, Conv1D, Conv2D, Reshape, Lambda
from keras.layers import GlobalMaxPooling1D, MaxPool2D, MaxPool1D, GlobalMaxPooling2D
from keras.layers import BatchNormalization, Embedding, Concatenate, Maximum, Add

from matchernet.model.feature_extraction import image_xception
from matchernet.model.iterator import Iterator, ModelIterator
from matchernet.model.metrics import generate_metrics
from matchernet.model.data import data_load, normalize

# define directories
data_dir = "data/"


def image_load():
    x_train_img_path = dataset_train["img"].values
    x_test_img_path = dataset_test["img"].values

    x_train_img = data_load(x_train_img_path)
    # no normalize for training images, since they will be augmented and normalized during training

    x_test_img = data_load(x_test_img_path)
    # normalize
    x_test_img = normalize(x_test_img)

    return x_train_img, x_test_img


dataset_train = pd.read_csv(os.path.join(data_dir, "dataset_train.csv"))
dataset_test = pd.read_csv(os.path.join(data_dir, "dataset_test.csv"))

x_train_img, x_test_img = image_load()

y_train = dataset_train["target"].values
y_test = dataset_test["target"].values

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

input_img = Input(shape=(128, 128, 3), name="input_tensor")

xcp = image_xception(input_img)

feature = Concatenate()([xcp])
feature = Dense(256, activation='relu', name="last_dense")(feature)

feature = Dropout(0.5)(feature)
feature = Dense(y_train.shape[1], activation='softmax', name="softmax")(feature)

model = Model([input_img], feature)

# optimization
model.compile(
    optimizer=Adam(lr=1e-4, decay=1e-6, amsgrad=True),
    loss=categorical_crossentropy,
    metrics=generate_metrics(class_num=11)
)

batch_size = 100
epochs = 20

training_iterator = Iterator(
    x_train_img,
    y_train,
    batch_size=batch_size)()

model_dir = "./model/{}".format(datetime.datetime.now().isoformat())
os.makedirs(model_dir, exist_ok=True)
chkpt = os.path.join(model_dir, 'model.{epoch:02d}_{loss:.4f}_{val_loss:.4f}.hdf5')

# train
model.fit_generator(
    training_iterator,
    steps_per_epoch=len(y_train) // batch_size,
    epochs=epochs,
    validation_data=([x_test_img], y_test),
    workers=6,
    use_multiprocessing=True,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto'),
               ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=1e-8),
               ModelCheckpoint(filepath=chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')])
