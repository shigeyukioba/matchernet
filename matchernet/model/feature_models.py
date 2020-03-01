import keras
import keras.layers as layer


def image_feature(inputs, output_size):
    """Image feature extraction by xception and conversion to fixed length

    Use the Xception V1 model (keras.applications.Xception) that can use
    the weights learned in ImageNet, implemented in keras.
    The accuracy of this model on ImageNet is
        top-1 validation accuracy: 0.790
        top-5 validation accuracy: 0.945
    At this time, by setting include_top = False, the entire coupling layer
    of the output is removed. In this way, only the feature extraction part
    can be used.
    This feature is made one-dimensional, and the output of output_size is
    obtained through the fully connected layer.
    A common use for flattening features is flatten, which converts
    (x, y, channels) dimensions to one (x * y * channels) dimension before
    entering the fully connected layer. However, GlobalAveragePooling2D adopts
    a method of performing average pooling of (x, y)-dimensional feature maps
    into one dimension and converting them into (channels) one-dimensional features.
    It is known that the accuracy can be improved while suppressing the number of
    parameters. In addition, this makes the input size arbitrary.
    Dense outputs the (batch_size, channels) to (batch_size, output_size) features.

    Args:
        inputs(numpy.array): (x, y, channels), Input image features
        output_size(int): Output feature size

    Returns:
        x: (batch_size, output_size), Image features
    """
    inception_feature = keras.applications.Xception(
        input_tensor=inputs,
        include_top=False,
        weights="imagenet"
    )
    feature = inception_feature.output
    x = layer.GlobalAveragePooling2D()(feature)
    x = layer.Dense(output_size, activation="relu")(x)

    return x