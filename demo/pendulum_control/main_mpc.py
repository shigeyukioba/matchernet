# -*- coding: utf-8 -*-
import numpy as np
from autograd import jacobian

from matchernet.state import StateMuSigma
from matchernet import utils
from matchernet import iLQG, MovieWriter, AnimGIFWriter
from pendulum import PendulumDynamics, PendulumCost, PendulumRenderer


class PendulumObservation(object):
    def __init__(self):
        # Jacobian calculation with automatic differentiation
        self.x = jacobian(self.value, 0)

    def value(self, x):
        return x


class EKFContinuousTime(object):
    def __init__(self, dynamics, Q, g, R, mu, Sigma):
        """
        Extended Kalman Filter for the continuous time.

        Parameters
        ----------
        dynamics : Dynamics
            Dyanmics of the control target
        Q : nd-array
            System noise covariance
        g : Fn
            Observation model
        R : nd-array
            Observation noise covariance
        mu : nd-array
            Initial state mu
        Sigma : nd-array
            Initial state covariance
        """
        self.dynamics = dynamics
        self.Q = Q
        self.g = g
        self.R = R
        self.state = StateMuSigma(mu, Sigma)
        
    def step_dynamics(self, u, dt):
        mu = self.state.data["mu"]
        Sigma = self.state.data["Sigma"]

        mu = mu + self.dynamics.value(mu, u) * dt
        A = self.dynamics.x(mu, u)
        F = utils.calc_matrix_F(A, dt)
        Sigma = F.T @ Sigma @ F + self.Q * dt
        
        self.state.data["mu"] = mu
        self.state.data["Sigma"] = Sigma

    def forward(self, y):
        mu = self.state.data["mu"]
        Sigma = self.state.data["Sigma"]
        
        z = y - self.g.value(mu)
        C = self.g.x(mu)
        S = C @ Sigma @ C.T + self.R
        
        SI = np.linalg.inv(S)
        
        K = Sigma @ C.T @ SI
        
        dmu = K @ z
        dSigma = -K @ C @ Sigma
        
        mu = self.state.data["mu"]
        Sigma = self.state.data["Sigma"]
        
        self.state.data["mu"] = mu + dmu
        self.state.data["Sigma"] = Sigma + dSigma


def main():
    np.random.rand(0)

    dt = 0.05
    dynamics = PendulumDynamics()
    cost = PendulumCost()
    
    # iLQG
    ilqg = iLQG(dynamics=dynamics, cost=cost, dt=dt)

    T = 80
    iter_max = 100
    
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
    movie = MovieWriter("pendulum_target0.mov", (256, 256), 30)
    for x, u in zip(x_list, u_list):
        image = renderer.render(x, u)
        
        image = (image * 255.0).astype(np.uint8)
        movie.add_frame(image)

    movie.close()
    
    
    
    # Record trajectory with EKF
    movie = MovieWriter("pendulum_ekf0.mov", (256, 256), 30)
    anim_gif = AnimGIFWriter("pendulum_ekf0.gif", 30)
    est_movie = MovieWriter("pendulum_ekf0_est.mov", (256, 256), 30)

    # Obervation model
    g = PendulumObservation()
    
    # System noise covariance
    Q = np.array([[0.01, 0.0],
                  [0.0, 0.01]], dtype=np.float32)
    
    # Observation noise
    R = np.array([[0.01, 0.0],
                  [0.0, 0.01]], dtype=np.float32)
    
    # Initial internal state
    mu0 = np.array([np.pi, 0.0], dtype=np.float32)
    Sigma0 = np.array([[0.001, 0.0],
                       [0.0, 0.001]], dtype=np.float32)

    # Initial real state
    x_real = np.random.multivariate_normal(mu0, Sigma0)
    
    # EKF
    ekf = EKFContinuousTime(dynamics, Q, g, R, mu0, Sigma0)
    
    # Initial control sequence for MPC
    u0 = np.zeros((T, 1), dtype=np.float32)

    # Calc optimal control trajectory
    x_list, u_list, K_list = ilqg.optimize(mu0,
                                           u0,
                                           T,
                                           iter_max=iter_max)
    
    for x_t, u_t, K_t in zip(x_list, u_list, K_list):
        # Estimated internal state posterior mu
        mu = ekf.state.data["mu"]
        
        # Calculate control signal
        u = u_t + K_t @ (mu - x_t)

        # Step dyanmmics in EKF
        ekf.step_dynamics(u, dt)

        # Update real state and observation
        xdot = dynamics.value(x_real, u)
        next_x_real = np.random.multivariate_normal(x_real + xdot * dt, Q * dt)
        y = np.random.multivariate_normal(g.value(x_real), R)

        # Estimate internal state given observation y
        ekf.forward(y)
        
        image = renderer.render(x_real, u)
        image = (image * 255.0).astype(np.uint8)

        estimated_image = renderer.render(mu, u)
        estimated_image = (estimated_image * 255.0).astype(np.uint8)
        
        movie.add_frame(image)
        anim_gif.add_frame(image)

        est_movie.add_frame(estimated_image)
        
        x_real = next_x_real
        
    movie.close()
    anim_gif.close()
    est_movie.close()

    

if __name__ == '__main__':
    main()
