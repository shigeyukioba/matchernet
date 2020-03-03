import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def random_erasing_augmentation(p=0.5,
                                s_l=0.02,
                                s_h=0.4,
                                r_1=0.3,
                                r_2=1 / 0.3,
                                v_l=0,
                                v_h=255):
    """Select a rectangular area from the image at random and mask it

    Specify the size and aspect ratio of the rectangle at random. The default value
    of each parameter used the value of the paper.

    reference: Zhun Zhong el al., 2017, Random Erasing Data Augmentation

    Args:
        p(float): Probability of using random erasing. default: 0.5
        s_l(float): Minimum ratio of mask area. default: 0.02
        s_h(float): Maximum ratio of mask area. default: 0.4
        r_1(float): Minimum value of aspect ratio of mask area. default: 0.3
        r_2(float): Maximum value of aspect ratio of mask area. default: 1/0.3
        v_l: default: 0
        v_h: default: 255

    Returns:
        ImageDataGenerator.preprocessing_function: Preprocessing functions
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


def image_generator(channel_shift_range=5.,
                    preprocessing_function=random_erasing_augmentation(v_l=0, v_h=255)):
    """Data augmentation and batch creation using keras standard generator

    Args:
        channel_shift_range(float): Range to shuffle channels randomly. default: 5.
        preprocessing_function: You can specify a function that takes a 3D numpy.ndarray
                                as input and has the same shape output.
                                default: get_random_eraser(v_l=0, v_h=255)

    Returns:
        ImageDataGenerator: A generator that generates image batches while performing
                            data augmentation.
    """
    return ImageDataGenerator(
        channel_shift_range=channel_shift_range,
        preprocessing_function=preprocessing_function
    )
