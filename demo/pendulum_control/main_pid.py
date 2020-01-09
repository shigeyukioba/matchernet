# -*- coding: utf-8 -*-
import numpy as np

from matchernet import MovieWriter, AnimGIFWriter
from pendulum import PendulumDynamics, PendulumRenderer


class PIDControl(object):
    def __init__(self, x0, x_target, dt, K_p, K_i, K_d, eta):
        self.x_last = x0
        self.x_target = x_target
        self.dt = dt

        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.eta = eta
        
        self.integral = 0.0

    def control(self, x):
        u_p = -self.K_p * (x - self.x_target)
        
        self.integral = self.integral * (1 - self.eta * self.dt) + self.eta * self.dt * (x - self.x_target)
        u_i = -self.K_i * self.integral
        
        u_d = -self.K_d * (x - self.x_last) / self.dt
        self.x_last = x
        return u_p + u_i + u_d


def main():
    np.random.rand(0)

    dt = 0.05
    dynamics = PendulumDynamics()
    
    # Initial state
    x0 = np.array([np.pi, 0.0], dtype=np.float32)
    x_target = np.array([0.0, 0.0], dtype=np.float32)
    
    # (81) (80) (80) (80)
    renderer = PendulumRenderer(image_width=256)

    # Record target trajectory
    movie = MovieWriter("out.mov", (256, 256), 30)
    
    dt = 0.02
    K_p = 2.3
    K_i = 0.8
    K_d = 0.22
    eta = 0.1
    
    pid_control = PIDControl(x0[0],
                             x_target[0],
                             dt,
                             K_p,
                             K_i,
                             K_d,
                             eta)
    
    x = np.copy(x0)
    
    for i in range(300):
        u = pid_control.control(x[0])
        u = np.array([u], dtype=np.float32)
        xdot = dynamics.value(x, u)
        x = x + xdot * dt
        image = renderer.render(x, u)
        
        image = (image * 255.0).astype(np.uint8)
        movie.add_frame(image)

    movie.close()


if __name__ == '__main__':
    main()
