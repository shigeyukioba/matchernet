# -*- coding: utf-8 -*-
import numpy as np
from autograd import jacobian

import brica
from brica import Component, VirtualTimeScheduler, Timing

from matchernet import MPCEnv
from matchernet.state import StateMuSigma
from matchernet.ekf import MatcherEKF
from pendulum import PendulumDynamics, PendulumCost

from matchernet import Bundle
from matchernet import utils


class BundleEKFWithController(Bundle):
    def __init__(self, name, dt, f, Q, mu, Sigma):
        super(BundleEKFWithController, self).__init__(name)        
        self.f = f
        self._initialize_control_params(dt)
        self._initialize_state(Q, mu, Sigma)

        self.update_component()        


    def __call__(self, inputs):
        for key in inputs:  # key is one of the matcher names
            if inputs[key] is not None:
                if "controller" not in key:
                    self.accept_feedback(inputs[key])

        for key in inputs:  # key is one of the matcher names
            if inputs[key] is not None:
                if "controller" in key:
                    u = inputs[key]["u"]
                    self.step_dynamics(u, self.dt)
                    
        self._countup()

        return {"state": self.state}

    def _initialize_control_params(self, dt):
        self.dt = dt

    def _countup(self):
        self.state.data["time_stamp"] = self.state.data["time_stamp"] + self.dt

    def _initialize_state(self, Q, mu, Sigma):
        self.state = StateMuSigma(mu, Sigma)
        self.state.data["Q"] = Q

    def accept_feedback(self, fbst):
        dmu = fbst.data["mu"]
        dSigma = fbst.data["Sigma"]
        mu = self.state.data["mu"]
        Sigma = self.state.data["Sigma"]

        self.state.data["mu"] = (mu + dmu).astype(np.float32)
        self.state.data["Sigma"] = (Sigma + dSigma).astype(np.float32)

    def step_dynamics(self, u, dt):
        mu = self.state.data["mu"]
        Sigma = self.state.data["Sigma"]
        Q = self.state.data["Q"]
        A = self.f.dx(mu)
        F = utils.calc_matrix_F(A, dt)
        mu = mu + self.dynamics.value(mu, u) * dt
        Sigma = dt * Q + F.T @ Sigma @ F
        Sigma = utils.regularize_cov_matrix(Sigma)
        
        self.state.data["mu"] = mu
        self.state.data["Sigma"] = Sigma
        # ["time_stamp"] is updated in the method self._countup()




class MPCEnvBundle(Bundle):
    """
    MPCEnv bundle class that communicates with matcher.
    """
    def __init__(self, env, R):
        """
        Arguments:
          env
            MPCEnv instance
          R
            observation noise
        """
        super(MPCEnvBundle, self).__init__("mpcenv_bundle")
        self.env = env

        x_dim = env.dynamics.x_dim
        mu = np.zeros((x_dim,), dtype=np.float32)
        Sigma = R
        self.state = StateMuSigma(mu, Sigma) 
        
        self.update_component()

    def __call__(self, inputs):
        u_dim = self.env.dynamics.u_dim
        u = np.zeros((u_dim,), dtype=np.float32)
        
        # TODO: u feedback 反映
        
        # Receive action from Matcher
        #u = feedback["u"]

        # Step environment with received action
        # Ignoring env rewards
        x, _ = self.env.step(u)

        self.state.data["mu"] = x
        self.state.data["time_stamp"] = self.state.data["time_stamp"] + self.env.dt
            
        # Send state to matcher
        return {
            "state" : self.state
        }



class PendulumObservation(object):
    def __init__(self):
        # Jacobian calculation with automatic differentiation
        self.x = jacobian(self.value, 0)

    def value(self, x):
        return x


def main():
    np.random.rand(0)

    dt = 0.05
    dynamics = PendulumDynamics()
    
    env = MPCEnv(dynamics, None, None, dt, use_visual_state=False)

    # Observation noise
    R = np.array([[0.01, 0.0],
                  [0.0, 0.01]], dtype=np.float32)
    mpcenv_b = MPCEnvBundle(env, R)

    # System noise covariance
    Q = np.array([[0.01, 0.0],
                  [0.0, 0.01]], dtype=np.float32)

    # Initial internal state
    mu0 = np.array([np.pi, 0.0], dtype=np.float32)
    Sigma0 = np.array([[0.001, 0.0],
                       [0.0, 0.001]], dtype=np.float32)
    
    ekf_b = BundleEKFWithController("ekf_contrller_bundle",
                                    dt,
                                    dynamics,
                                    Q,
                                    mu0,
                                    Sigma0)

    g0 = PendulumObservation()
    g1 = PendulumObservation()

    matcher = MatcherEKF("ekf_matcher", mpcenv_b, ekf_b, g0, g1)
    
    scheduler = VirtualTimeScheduler()
    
    timing0 = Timing(0, 1, 1)
    timing1 = Timing(1, 1, 1)
    
    scheduler.add_component(mpcenv_b.component, timing0)
    scheduler.add_component(ekf_b.component, timing0)
    scheduler.add_component(matcher.component, timing1)

    num_steps = 10
    
    for i in range(num_steps):
        print("Step {}/{}".format(i, num_steps))
        scheduler.step()
    
if __name__ == '__main__':
    main()
