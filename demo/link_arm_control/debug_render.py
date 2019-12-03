# -*- coding: utf-8 -*-
import numpy as np
from matchernet import MovieWriter, AnimGIFWriter

from link_arm import LinkArmDynamics, LinkArmRenderer

    
def concat_images(images):
    image_height = images[0].shape[0]
    spacer = np.ones([image_height, 1, 3], dtype=np.uint8) * 255
    images_with_spacers = []
    
    image_size = len(images)
    
    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size-1:
            # add 1 pixel space
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def check_renderer():
    eye_from0 = np.array([0.0, 0.0, -6.0], dtype=np.float32)
    eye_to0   = np.array([0.0, 0.0, 0.0],  dtype=np.float32)

    eye_from1 = np.array([0.0, -4.0, -4.0], dtype=np.float32)
    eye_to1   = np.array([0.0, -2.0, 0.0], dtype=np.float32)

    eye_from2 = np.array([6.0, 0.0, 0.0], dtype=np.float32)
    eye_to2   = np.array([0.0, 0.0, 0.0], dtype=np.float32)    

    buffer_width = 128
    renderer0 = LinkArmRenderer(eye_from0, eye_to0, buffer_width)
    renderer1 = LinkArmRenderer(eye_from1, eye_to1, buffer_width)
    renderer2 = LinkArmRenderer(eye_from2, eye_to2, buffer_width)

    dynamics = LinkArmDynamics(dt=0.01)

    x = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    frame_size = 180

    movie = MovieWriter("link_arm0.mov", (128*3+2, 128), 60)
    anim_gif = AnimGIFWriter("link_arm0.gif", 60)

    for i in range(frame_size):
        u = np.zeros((2,), dtype=np.float32)
        x = dynamics.value(x, u)
        image0 = renderer0.render(x)
        image1 = renderer1.render(x)
        image2 = renderer2.render(x)

        image = concat_images([image0, image1, image2])

        movie.add_frame(image)
        anim_gif.add_frame(image)        

    movie.close()
    anim_gif.close()


check_renderer()
