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
        self._initialize_state(mu, Sigma)

        self.Q = Q
        self.time_stamp = 0.0

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

        results = {
            "mu": self.state.data["mu"],
            "Sigma": self.state.data["Sigma"],
            "time_stamp": self.time_stamp
        }
        return {"state": results}

    def _initialize_control_params(self, dt):
        self.dt = dt

    def _countup(self):
        self.time_stamp += self.dt

    def _initialize_state(self, mu, Sigma):
        self.state = StateMuSigma(mu, Sigma)

    def accept_feedback(self, fbst):
        dmu = fbst["mu"]
        dSigma = fbst["Sigma"]
        mu = self.state.data["mu"]
        Sigma = self.state.data["Sigma"]

        self.state.data["mu"] = (mu + dmu).astype(np.float32)
        self.state.data["Sigma"] = (Sigma + dSigma).astype(np.float32)

    def step_dynamics(self, u, dt):
        mu = self.state.data["mu"]
        Sigma = self.state.data["Sigma"]
        Q = self.Q
        A = self.f.x(mu, u)
        F = utils.calc_matrix_F(A, dt)
        mu = mu + self.f.value(mu, u) * dt
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
        self.R = R
        self.time_stamp = 0.0
        self.update_component()

    def __call__(self, inputs):
        u_dim = self.env.dynamics.u_dim

        # Receive action from Matcher
        if "matcher_controller" in inputs.keys() and inputs["matcher_controller"] is not None:
            u = inputs["matcher_controller"]["u"]
        else:
            u = np.zeros((u_dim, ), dtype=np.float32)

        # Step environment with received action
        # Ignoring env rewards
        x, _ = self.env.step(u)
        
        self.time_stamp += self.env.dt

        results = {}
        results["mu"] = x
        results["Sigma"] = self.R
        results["time_stamp"] = self.time_stamp

        # Send state to matcher
        return {"state": results}


class MatcherController(object):
    def __init__(self, mpcenv_bundle, ekf_bundle, plan_bundle):
        self.name = "matcher_controller"
        self.results = {}

        self.mpcenv_bundle = mpcenv_bundle
        self.ekf_bundle = ekf_bundle
        self.plan_bundle = plan_bundle
        
        self.results[self.mpcenv_bundle.name] = {}
        self.results[self.ekf_bundle.name] = {}
        # plan_bundle宛てのresultsは無いので注意

        self.update_component()

    def update_component(self):
        component = Component(self)

        # MatcherController -> MPCEnvBundle
        component.make_out_port(self.mpcenv_bundle.name)
        self.mpcenv_bundle.component.make_in_port(self.name)  # -> send u to MPCEnv
        brica.connect(component, self.mpcenv_bundle.name,
                      self.mpcenv_bundle.component, self.name)

        # BundleEKFWithController <-> MatcherController
        component.make_in_port(self.ekf_bundle.name)  # receive state from BundleEKF
        component.make_out_port(self.ekf_bundle.name)  # send u to BundleEKF
        self.ekf_bundle.component.make_in_port(self.name)

        brica.connect(self.ekf_bundle.component, "state", 
                      component, self.ekf_bundle.name)
        brica.connect(component, self.ekf_bundle.name,
                      self.ekf_bundle.component, self.name)

        # BundlePlan -> MatcherController
        component.make_in_port(self.plan_bundle.name)  # receive state from BundlePlan
        brica.connect(self.plan_bundle.component, "state", 
                      component, self.plan_bundle.name)
        
        self.component = component

    def __call__(self, inputs):
        self.update(inputs)

        return self.results

    def update(self, inputs):
        ekf_state = inputs[self.ekf_bundle.name]
        plan_state = inputs[self.plan_bundle.name]
        
        x_plan = plan_state["x"]
        u_plan = plan_state["u"]
        K_plan = plan_state["K"]

        mu = ekf_state["mu"]

        u = u_plan + K_plan @ (mu - x_plan)

        self.results[self.mpcenv_bundle.name]["u"] = u
        self.results[self.ekf_bundle.name]["u"] = u


class Plan(object):
    def __init__(self):
        pass


class MatcherILQR(object):
    def __init__(self, ekf_bundle, plan_bundle):
        self.name = "matcher_ilqr"
        self.results = {}

        self.ekf_bundle = ekf_bundle
        self.plan_bundle = plan_bundle
        self.bundles = (ekf_bundle, plan_bundle)

        self.results[self.plan_bundle.name] = {}

        self.update_component()

    def update_component(self):
        component = Component(self)

        # BundleEKFWithController -> MatcherILQR
        component.make_in_port(self.ekf_bundle.name)  # receive state from BundleEKF
        brica.connect(self.ekf_bundle.component, "state", component,
                      self.ekf_bundle.name)

        # MatcherController -> BundlePlan
        component.make_out_port(self.plan_bundle.name)  # send Plan to BundlePlan
        self.plan_bundle.component.make_in_port(self.name)
        brica.connect(component, self.plan_bundle.name,
                      self.plan_bundle.component, self.name)
        
        self.component = component

    def __call__(self, inputs):
        self.update(inputs)

        return self.results

    def update(self, inputs):
        ekf_state = inputs[self.ekf_bundle.name]
        
        mu = ekf_state["mu"]
        plan = Plan()
        
        self.results[self.plan_bundle.name]["plan"] = plan


class BundlePlan(Bundle):
    def __init__(self, x_dim, u_dim):
        super(BundlePlan, self).__init__("plan_bundle")
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.update_component()

    def __call__(self, inputs):
        # TODO: 実装中
        x = np.zeros((self.x_dim,), dtype=np.float32)
        u = np.zeros((self.u_dim,), dtype=np.float32)
        K = np.zeros((self.u_dim, self.x_dim), dtype=np.float32)
        
        results = {
            "x": x,
            "u": u,
            "K": K
        }
        return {"state": results}


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
    R = np.array([[0.01, 0.0], [0.0, 0.01]], dtype=np.float32)
    mpcenv_b = MPCEnvBundle(env, R)

    # System noise covariance
    Q = np.array([[0.01, 0.0], [0.0, 0.01]], dtype=np.float32)

    # Initial internal state
    mu0 = np.array([np.pi, 0.0], dtype=np.float32)
    Sigma0 = np.array([[0.001, 0.0], [0.0, 0.001]], dtype=np.float32)

    # EKF Controller Bundle
    ekf_b = BundleEKFWithController("ekf_contrller_bundle", dt, dynamics, Q,
                                    mu0, Sigma0)

    # Plan Bundle
    plan_b = BundlePlan(dynamics.x_dim, dynamics.u_dim)

    # Controller Matcher
    controller_m = MatcherController(mpcenv_b, ekf_b, plan_b)

    g0 = PendulumObservation()
    g1 = PendulumObservation()

    # EKF Matcher
    ekf_m = MatcherEKF("ekf_matcher", mpcenv_b, ekf_b, g0, g1)

    # ILQR Matcher
    ilqr_m = MatcherILQR(ekf_b, plan_b)

    scheduler = VirtualTimeScheduler()

    timing0 = Timing(0, 1, 1)
    timing1 = Timing(1, 1, 1)

    # TODO: iLQRのインターバル設定
    scheduler.add_component(mpcenv_b.component, timing0)
    scheduler.add_component(ekf_b.component, timing0)
    scheduler.add_component(ekf_m.component, timing1)
    scheduler.add_component(controller_m.component, timing1)
    scheduler.add_component(plan_b.component, timing0)
    scheduler.add_component(ilqr_m.component, timing1)

    num_steps = 10

    for i in range(num_steps):
        print("Step {}/{}".format(i, num_steps))
        scheduler.step()


if __name__ == '__main__':
    main()
