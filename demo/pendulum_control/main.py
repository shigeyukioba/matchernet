# -*- coding: utf-8 -*-
import numpy as np

from matchernet import iLQG, MovieWriter, AnimGIFWriter
from pendulum import PendulumDynamics, PendulumCost, PendulumRenderer


def main():
    dynamics = PendulumDynamics(dt=0.05)
    cost = PendulumCost()
    
    ilqg = iLQG(dynamics=dynamics, cost=cost)
    
    T = 80
    iter_max = 50
    
    # Initial state
    x0 = np.array([np.pi, 0.0], dtype=np.float32)
    
    # Initial control sequence
    u0 = np.zeros((T, 1), dtype=np.float32)
    
    x_list, u_list, K_list = ilqg.optimize(x0,
                                           u0,
                                           T,
                                           iter_max=iter_max)
    # (81) (80) (80) (80)
    renderer = PendulumRenderer(image_width=256)

    # Record target trajectory
    movie = MovieWriter("pendulum_ilqg_trajectory0.mov", (256, 256), 30)
    for x, u in zip(x_list, u_list):
        image = renderer.render(x, u)
        
        image = (image * 255.0).astype(np.uint8)
        movie.add_frame(image)

    movie.close()

    # System noise covariance
    Q = np.array([[0.01, 0.0],
                  [0.0, 0.01]], dtype=np.float32)

    # Record real trajectory
    movie = MovieWriter("pendulum_ilqg_real0.mov", (256, 256), 30)
    anim_gif = AnimGIFWriter("pendulum_ilqg_real0.gif", 30)

    # Randomize initial state
    initial_cov = np.array([[0.2, 0.0],
                            [0.0, 0.1]], dtype=np.float32)
    x =  np.random.multivariate_normal(x0, initial_cov)
    
    for x_t, u_t, K_t in zip(x_list, u_list, K_list):
        u = u_t + K_t @ (x - x_t)

        # Add system noise to dynamics
        system_noise = np.random.multivariate_normal(np.zeros_like(x), Q * dynamics.dt)
        next_x = dynamics.value(x, u) + system_noise
        
        image = renderer.render(x, u)
        image = (image * 255.0).astype(np.uint8)
        
        movie.add_frame(image)
        anim_gif.add_frame(image)
        
        x = next_x
        
    movie.close()
    anim_gif.close()

    

if __name__ == '__main__':
    main()
