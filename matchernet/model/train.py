import pandas as pd
import keras

from matchernet.model import model
from matchernet.model import iterator

if __name__ == "__main__":
    train_data = pd.read_csv("data/dataset_train.csv", index_col=0)
    test_data = pd.read_csv("data/dataset_test.csv", index_col=0)

    model = model.MultiModalClassifier(
        inputs_df=train_data,
        num_classes=11,
        target_column="target"
    )()

    model.compile(
        optimizer=keras.optimizers.Adam(
            lr=1e-4,
            decay=1e-6,
            amsgrad=True
        ),
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    batch_size = 30
    epochs = 1

    training_generator = iterator.MultiModalIterator(
        train_data,
        target_column="target",
        train=True,
        model_type="multiclass",
        batch_size=batch_size,
        shuffle=True)()

    val_generator = iterator.MultiModalIterator(
        test_data,
        target_column="target",
        train=False,
        model_type="multiclass",
        batch_size=batch_size,
        shuffle=False)()

    model.fit_generator(
        training_generator,
        steps_per_epoch=len(train_data) // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(test_data) // batch_size,
        workers=6,
        use_multiprocessing=True
    )
