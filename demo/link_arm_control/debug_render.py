# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

from matchernet import MPCEnv, MovieWriter, AnimGIFWriter
from link_arm import LinkArmDynamics, LinkArmRenderer


def concat_images(images):
    """ Concatenate image into one image """
    image_height = images[0].shape[0]
    spacer = np.ones([image_height, 1, 3], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add 1 pixel space
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def save_images(index, images, state):
    for angle in range(images.shape[0]):
        data_dir = './data/angle{}'.format(angle)
        # if os.path.exists(data_dir):
        #     continue
        # else:
        #     os.makedirs(data_dir)
        print(data_dir)
        numpy_image = np.asarray(images[angle])
        np.savez('./data/angle{}/image{}'.format(angle, index), img=numpy_image, y=state)


def main():
    # Prepare renderers with multiple camera angles.
    # Camera angles are defined with eye origin position and eye target position.
    eye_from0 = np.array([0.0, 0.0, -6.0], dtype=np.float32)
    eye_to0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    eye_from1 = np.array([0.0, -4.0, -4.0], dtype=np.float32)
    eye_to1 = np.array([0.0, -2.0, 0.0], dtype=np.float32)

    eye_from2 = np.array([6.0, 0.0, 0.0], dtype=np.float32)
    eye_to2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    image_width = 128
    renderer0 = LinkArmRenderer(eye_from0, eye_to0, image_width)
    renderer1 = LinkArmRenderer(eye_from1, eye_to1, image_width)
    renderer2 = LinkArmRenderer(eye_from2, eye_to2, image_width)

    renderer = [renderer0, renderer1, renderer2]

    # Dynamics
    dynamics = LinkArmDynamics()

    # MPC Environment
    dt = 0.01
    env = MPCEnv(dynamics, renderer, reward_system=None, dt=dt, use_visual_state=True)

    # Initial state
    x_init = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    env.reset(x_init)

    frame_size = 180

    movie = MovieWriter("link_arm0.mov", (128 * 3 + 2, 128), 60)
    anim_gif = AnimGIFWriter("link_arm0.gif", 60)

    for i in range(frame_size):
        # Zero torque applied
        u = np.zeros((2,), dtype=np.float32)

        # Step environment
        state, _ = env.step(u)
        # state has shape (3, 128, 128, 3) = (angle_size, h, w, ch), dtype=uint8 (0~255)

        # Concatenate images into one
        save_images(i, state, env.x)
        # image = concat_images(state)
        #
        # movie.add_frame(image)
        # anim_gif.add_frame(image)
        # if i > 30:
        #     break

    movie.close()
    anim_gif.close()


main()
