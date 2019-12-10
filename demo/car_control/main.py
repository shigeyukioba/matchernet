# -*- coding: utf-8 -*-
import numpy as np

from matchernet import iLQG, MovieWriter, AnimGIFWriter
from car import CarDynamics, CarCost, CarRenderer, CarObstacle

import cv2


def render_obstacles(image, obstacles):
    image_width = image.shape[1]
    render_scale = image_width / 2.0
    render_offset = image_width / 2.0
    
    for obstacle in obstacles:
        rx = int(obstacle.pos[0] * render_scale + render_offset)
        ry = int(obstacle.pos[1] * render_scale + render_offset)
        if obstacle.is_good:
            reward_color = (1,0,0)
        else:
            reward_color = (0,0,0)
        radius = 0.1
        image = cv2.circle(image,
                           (rx, ry), int(radius * render_scale),
                           reward_color, -1)
    return image
    

def main():
    dynamics = CarDynamics(dt=0.03)

    renderer = CarRenderer(image_width=256)

    obstacles = []
    obstacle0 = CarObstacle(pos=np.array([0.5, 0.0], dtype=np.float32),
                            is_good=False)
    obstacles.append(obstacle0)
    obstacle1 = CarObstacle(pos=np.array([0.5, 0.3], dtype=np.float32),
                            is_good=True)
    obstacles.append(obstacle1)
    cost = CarCost(obstacles)
    
    ilqg = iLQG(dynamics=dynamics, cost=cost)

    T = 120
    
    # Initial state
    x0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Initial control sequence
    u0 = np.zeros((T, 2), dtype=np.float32)

    movie = MovieWriter("car_ilqg0.mov", (256, 256), 60)
    anim_gif = AnimGIFWriter("car_ilqg0.gif", 60)
    
    x_list, u_list, K_list = ilqg.optimize(x0,
                                           u0,
                                           T,
                                           iter_max=30)

    x = x0
    
    for x_t, u_t, K_t in zip(x_list, u_list, K_list):
        u = u_t + K_t @ (x - x_t)
        next_x = dynamics.value(x, u)
        x = next_x
        
        image = renderer.render(x, u)
        render_obstacles(image, obstacles)
        
        image = (image * 255.0).astype(np.uint8)
        movie.add_frame(image)
        anim_gif.add_frame(image)
        
    movie.close()
    anim_gif.close()

if __name__ == '__main__':
    main()
