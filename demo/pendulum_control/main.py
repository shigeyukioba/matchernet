# -*- coding: utf-8 -*-
import numpy as np

from matchernet import iLQG, MovieWriter
from pendulum import PendulumDynamics, PendulumCost, PendulumRenderer


def main():
    dynamics = PendulumDynamics(dt=0.05)
    cost = PendulumCost()
    
    ilqg = iLQG(dynamics=dynamics, cost=cost)

    T = 80
    x0 = np.array([np.pi, 0.0], dtype=np.float32)
    x_list, u_list, k_list, K_list = ilqg.optimize(x0,
                                                   T,
                                                   iter_max=20)
    # (81) (80) (80) (80)
    renderer = PendulumRenderer()

    # Record target trajectory
    movie = MovieWriter("pendulum_ilqg_trajectory0.mov", (256, 256), 30)
    for x, u in zip(x_list, u_list):
        image = np.ones((256, 256, 3), dtype=np.float32)
        renderer.render(image, x, u)

        image = (image * 255.0).astype(np.uint8)
        movie.add_frame(image)

    movie.close()

    Q = np.array([[0.01, 0.0],
                  [0.0, 0.01]], dtype=np.float32)

    # Record real trajectory
    movie = MovieWriter("pendulum_ilqg_real0.mov", (256, 256), 30)
    x = np.copy(x0)
    for x_t, u_t, k_t, K_t in zip(x_list, u_list, k_list, K_list):
        u = u_t + k_t + K_t @ (x - x_t)
        
        system_noise = np.random.multivariate_normal(np.zeros_like(x), Q * dynamics.dt)
        next_x = dynamics.value(x, u) + system_noise
        
        image = np.ones((256, 256, 3), dtype=np.float32)
        renderer.render(image, x, u)
        
        image = (image * 255.0).astype(np.uint8)
        movie.add_frame(image)
        x = next_x
    movie.close()
    
    

if __name__ == '__main__':
    main()
