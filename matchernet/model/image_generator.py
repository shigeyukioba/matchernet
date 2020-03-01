import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def random_erasing_augmentation(p=0.5,
                                s_l=0.02,
                                s_h=0.4,
                                r_1=0.3,
                                r_2=1 / 0.3,
                                v_l=0,
                                v_h=255):
    """画像から矩形領域をランダムで選びマスクする

    矩形の大きさ、アスペクト比をランダムに指定する。各パラメータの default 値は、
    論文の値を使用した。

    参考：Zhun Zhong el al., 2017, Random Erasing Data Augmentation

    Args:
        p(float): random erasing を使用する確率 default: 0.5
        s_l(float): マスク領域の最小比率 default: 0.02
        s_h(float): マスク領域の最大比率 default: 0.4
        r_1(float): マスク領域のアスペクト比の最小値 default: 0.3
        r_2(float): マスク領域のアスペクト比の最大値 default: 1/0.3
        v_l: default: 0
        v_h: default: 255

    Returns:
        ImageDataGenerator.preprocessing_function: 前処理の関数
    """

    def eraser(input_image):
        if np.random.rand() > p:
            return input_image

        while True:
            image_h, image_w, _ = input_image.shape
            s = np.random.uniform(s_l, s_h) * image_h * image_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, image_w)
            top = np.random.randint(0, image_h)

            if left + w <= image_w and top + h <= image_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_image[top:top + h, left:left + w, :] = c

        return input_image

    return eraser


def image_generator(rotation_range=180,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=10,
                    zoom_range=0.3,
                    horizontal_flip=True,
                    vertical_flip=True,
                    channel_shift_range=5.,
                    preprocessing_function=random_erasing_augmentation(v_l=0, v_h=255)):
    """keras 標準の generator を使用した data augmentation とバッチ作成

    Args:
        rotation_range(int): 画像をランダムに回転させる回転範囲 default: 180
        width_shift_range(float): ランダムに水平シフトする範囲 default: 0.2
        height_shift_range(float): ランダムに垂直シフトする範囲 default: 0.2
        shear_range(float): シアー強度という、斜めに歪ませるような変換をかける強さ default: 10
        zoom_range(float): 画像内の物体のみを拡大する default: 0.3
        horizontal_flip(bool): 水平方向にランダムに反転する default: True
        vertical_flip(bool): 垂直方向にランダム反転する default: True
        channel_shift_range(float): ランダムにチャンネルをシャッフルする範囲 default: 5.
        preprocessing_function: 3次元の numpy.ndarray を入力にとり、同じ shape の出力を持つ
                                関数を指定できる default: get_random_eraser(v_l=0, v_h=255)

    Returns:
        ImageDataGenerator: data augmentation しつつ画像バッチを生成する generator
    """
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        channel_shift_range=channel_shift_range,
        preprocessing_function=preprocessing_function
    )
