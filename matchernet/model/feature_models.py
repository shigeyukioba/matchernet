import keras
import keras.layers as layer


def image_feature(inputs, output_size):
    """xception による画像特徴量抽出と固定長への変換

    keras に実装されている、ImageNet で学習した重みを利用可能な Xception V1 のモデル
    (keras.applications.Xception) を使用する。
    このモデルの ImageNet での精度は、
        top-1 validation accuracy: 0.790
        top-5 validation accuracy: 0.945
    である。
    この際、include_top=False とすることで、出力の全結合層を外している。こうすることで、
    特徴抽出の部分のみ利用可能になる。
    この特徴量を1次元化し全結合層を通して output_size の出力を得る。
    特徴量の1次元化によく使われるのは flatten で、これは全結合層に入力する前に (x, y, channels)
    次元の特徴量を (x*y*channels) の1次元に変換するものだが、GlobalAveragePooling2D では
    (x, y) 次元の特徴マップを1次元に average pooling し、(channels) の1次元特徴に変換する
    方式を採用した。これは、パラメータ数を抑制しつつ精度も出せることが知られている。また、
    これにより、入力サイズが任意になっている。
    Dense で (batch_size, channels) から (batch_size, output_size) の特徴量を出力する。

    Args:
        inputs(numpy.array): (x, y, channels), 入力画像特徴
        output_size(int): 出力する特徴量のサイズ

    Returns:
        x: (batch_size, output_size), 画像特徴量
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