# -*- coding: utf-8 -*-
import numpy as np
import brica
from brica import Component

from matchernet.control.ilqg import iLQG
from matchernet.matchernet import Bundle


class MatcherController(object):
    def __init__(self, name, mpcenv_bundle, ekf_bundle, plan_bundle):
        self.name = name
        self.results = {}

        self.mpcenv_bundle = mpcenv_bundle
        self.ekf_bundle = ekf_bundle
        self.plan_bundle = plan_bundle
        
        self.results[self.mpcenv_bundle.name] = {}
        self.results[self.ekf_bundle.name] = {}
        # Note: there is no results entry for plan_bundle

        # TODO: How do we reset this integral value for PID?
        self.integral = 0.0
        self.x_last = None

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
        K_i_plan = plan_state["K_i"]
        K_d_plan = plan_state["K_d"]
        eta = plan_state["eta"]
        dt = plan_state["dt"]
        plan_time_stamp = plan_state["time_stamp"]
        plan_time_id = plan_state["time_id"]

        mu = ekf_state["mu"]
        ekf_time_stamp = ekf_state["time_stamp"]
        ekf_time_id = ekf_state["time_id"]

        # PID control
        u_p = u_plan + K_plan @ (mu - x_plan)
        self.integral = self.integral * (1 - eta * dt) + eta * dt * (mu - x_plan)
        u_i = -K_i_plan @ self.integral
        if self.x_last is None:
            self.x_last = np.copy(mu)
        u_d = -K_d_plan @ (mu - self.x_last) / dt
        u = u_p + u_i + u_d
        
        self.x_last = np.copy(mu)
        
        self.results[self.mpcenv_bundle.name]["u"] = u
        self.results[self.ekf_bundle.name]["u"] = u


class Plan(object):
    def __init__(self, x_list, u_list, K_list, time_stamp, time_id):
        self.x_list = x_list
        self.u_list = u_list
        self.K_list = K_list
        self.time_stamp = time_stamp
        self.time_id = time_id


class MatcherILQR(object):
    def __init__(self, name, ekf_bundle, plan_bundle,
                 dynamics, cost, dt, T, iter_max):
        self.name = name
        self.results = {}

        self.ekf_bundle = ekf_bundle
        self.plan_bundle = plan_bundle
        self.bundles = (ekf_bundle, plan_bundle)

        self.results[self.plan_bundle.name] = {}

        self.ilqg = iLQG(dynamics=dynamics, cost=cost, dt=dt)
        
        self.T = T
        self.iter_max = iter_max

        self.last_u_list = None
        self.last_time_id = 0
        
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
        time_stamp = ekf_state["time_stamp"]
        time_id = ekf_state["time_id"]

        # Initial control sequence for MPC
        u_dim = self.ilqg.dynamics.u_dim
        u0 = np.zeros((self.T, u_dim), dtype=np.float32)
        if self.last_u_list is not None:
            # Set initial control signals by copying last control signals.
            advance_time_id = time_id - self.last_time_id
            u0[:self.T-advance_time_id,:] = self.last_u_list[advance_time_id:,:]

        x_list, u_list, K_list = self.ilqg.optimize(mu,
                                                    u0,
                                                    self.T,
                                                    self.iter_max)
        plan = Plan(x_list, u_list, K_list, time_stamp, time_id)
        self.results[self.plan_bundle.name]["plan"] = plan
        
        self.last_u_list = u_list
        self.last_time_id = time_id


class BundlePlan(Bundle):
    """
    Plan sending class for iLQR control.
    """
    def __init__(self, name, x_dim, u_dim, dt, control_T, plan_src_name):
        super(BundlePlan, self).__init__(name)
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.dt = dt
        self.control_T = control_T
        self.plan_src_name = plan_src_name
        
        self.latest_plan = None
        self.index_in_plan = 0
        
        self.update_component()

    def __call__(self, inputs):
        if self.plan_src_name in inputs.keys() and inputs[self.plan_src_name] is not None:
            plan = inputs[self.plan_src_name]["plan"]
            if (self.latest_plan is None) or (self.latest_plan.time_id != plan.time_id):
                self.latest_plan = plan
                self.index_in_plan = self.control_T - 1
                # Note: To make the timestamp equal to that from EKFBundle, we need to start
                # index_in_plan as 'control_T + 1', but we are now using 'control_T - 1'
        
        if self.latest_plan is not None:
            x = self.latest_plan.x_list[self.index_in_plan]
            u = self.latest_plan.u_list[self.index_in_plan]
            K = self.latest_plan.K_list[self.index_in_plan]
            time_stamp = self.latest_plan.time_stamp + self.dt * self.index_in_plan
            time_id = self.latest_plan.time_id + self.index_in_plan
            self.index_in_plan += 1
        else:
            x = np.zeros((self.x_dim,), dtype=np.float32)
            u = np.zeros((self.u_dim,), dtype=np.float32)
            K = np.zeros((self.u_dim, self.x_dim), dtype=np.float32)
            time_stamp = 0
            time_id = 0

        # Compatibility for PID controller
        K_i = np.zeros((self.u_dim, self.x_dim), dtype=np.float32)
        K_d = np.zeros((self.u_dim, self.x_dim), dtype=np.float32)
        eta = 0.0
        
        results = {
            "x": x,
            "u": u,
            "K": K,
            "K_i": K_i,
            "K_d": K_d,
            "eta": eta,
            "dt": self.dt,
            "time_stamp": time_stamp,
            "time_id": time_id
        }
        return {"state": results}


class BundleFixedPlan(Bundle):
    """
    Plan sending class for PID control.
    """
    def __init__(self, name, x_dim, u_dim, dt, x_target, K, K_i, K_d, eta):
        super(BundleFixedPlan, self).__init__(name)
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.x_target = x_target
        self.K = K
        self.K_i = K_i
        self.K_d = K_d
        self.eta = eta
        self.dt = dt

        self.time_stamp = 0.0
        self.time_id = 0
        
        self.update_component()

    def __call__(self, inputs):
        # Compatibility for iLQR controller
        u = np.zeros((self.u_dim,), dtype=np.float32)
        
        results = {
            "x": self.x_target,
            "u": u,
            "K": -self.K, # Sign is inverted for the compatibility with iLQR
            "K_i": self.K_i,
            "K_d": self.K_d,
            "eta": self.eta,
            "dt": self.dt,
            "time_stamp": self.time_stamp,
            "time_id": self.time_id
        }

        self.time_stamp += self.dt
        self.time_id += 1
        
        return {"state": results}


class MPCEnvBundle(Bundle):
    """
    MPCEnv bundle class that communicates with matcher.
    """

    def __init__(self, name, env, R, control_src_name, debug_recorder=None):
        """
        Arguments:
          env
            MPCEnv instance
          R
            Observation noise
          control_src_name
            Name of controller src node (Controller Matcher)
        """
        super(MPCEnvBundle, self).__init__(name)
        self.env = env
        self.R = R
        self.time_stamp = 0.0
        self.time_id = 0
        self.control_src_name = control_src_name
        self.debug_recorder = debug_recorder
        
        self.update_component()

    def __call__(self, inputs):
        u_dim = self.env.u_dim

        # Receive action from Controller Matcher
        if self.control_src_name in inputs.keys() and inputs[self.control_src_name] is not None:
            u = inputs[self.control_src_name]["u"]
        else:
            u = np.zeros((u_dim,), dtype=np.float32)

        # Step environment with received action
        # Ignoring env rewards
        x, _ = self.env.step(u)
        
        results = {}
        results["mu"] = x
        results["Sigma"] = self.R
        results["time_stamp"] = self.time_stamp
        results["time_id"] = self.time_id

        if self.debug_recorder is not None:
            self.debug_recorder.record(x, u)

        self.time_stamp += self.env.dt
        self.time_id += 1

        # Send state to matcher
        return {"state": results}
