# -*- coding: utf-8 -*-
import numpy as np

import brica
from brica import VirtualTimeScheduler, Timing

from matchernet import MPCEnvBundle, MatcherController, Plan, MatcherILQR, BundlePlan, BundleEKFWithController, MatcherEKF, MPCEnv, MovieWriter, AnimGIFWriter

#from vrep_control import VREPDynamics, VREPCost

from matchernet import Dynamics
from autograd import jacobian, grad
from vrep_env import VREPEnv


class VREPDynamics(Dynamics):
    def __init__(self):
        super(VREPDynamics, self).__init__()
        
        self.x = jacobian(self.value, 0)
        self.u = jacobian(self.value, 1)

    def value(self, x, u):
        # TODO: 未実装
        return x

    @property
    def x_dim(self):
        return 12

    @property
    def u_dim(self):
        return 6


class VREPCost(object):
    def __init__(self):
        self.x  = grad(self.value, 0)
        self.u  = grad(self.value, 1)
        self.xx = jacobian(self.x, 0)
        self.uu = jacobian(self.u, 1)
        self.ux = jacobian(self.u, 0)

    def clone(self):
        return self

    def apply_state(self, x, t):
        pass

    def value(self, x, u, t):
        # TODO: 未実装
        return 0.0


class Observation(object):
    def __init__(self):
        # Jacobian calculation with automatic differentiation
        self.x = jacobian(self.value, 0)
        
    def value(self, x):
        return x
    

def main():
    np.random.rand(0)

    dt = 0.005
    dynamics = VREPDynamics()
    cost = VREPCost()
    T = 30 # MPC Horizon
    control_T = 10 # Plan update interval for receding horizon
    iter_max = 20
    num_steps = 300

    # Component names
    ekf_controller_bundle_name = "ekf_contrller_bundle"
    ekf_matcher_name = "ekf_matcher"
    plan_bundle_name = "plan_bundle"
    controller_matcher_name = "controller_matcher"
    ilqr_matcher_name = "ilqr_matcher"
    mpcenv_bundle_name = "mpc_env_bundle"

    # TODO:
    # Initial state
    x0 = np.zeros((12,), dtype=np.float32)

    # Initial internal state
    mu0 = np.zeros((12,), dtype=np.float32)
    Sigma0 = np.eye(12, dtype=np.float32) * 0.001
    
    # System noise covariance
    Q = np.eye(12, dtype=np.float32) * 0.0001

    # Observation noise
    R = np.eye(12, dtype=np.float32) * 0.0001
    
    # MPCEnv Bundle
    env = VREPEnv()
    default_angles = np.array([-3.48837466e-06,
                               2.75779843e+00,
                               2.70459056e+00,
                               -1.57118285e+00,
                               1.70024168e-02,
                               3.07200003e+00])
    q0 = default_angles
    env.reset_angles(q0)
    
    mpcenv_b = MPCEnvBundle(mpcenv_bundle_name, env, R,
                            controller_matcher_name)

    # EKF Controller Bundle
    ekf_b = BundleEKFWithController(ekf_controller_bundle_name, dt, dynamics, Q,
                                    mu0, Sigma0, controller_matcher_name)
    
    # Plan Bundle
    plan_b = BundlePlan(plan_bundle_name, dynamics.x_dim, dynamics.u_dim, dt, control_T,
                        ilqr_matcher_name)

    # Controller Matcher
    controller_m = MatcherController(controller_matcher_name, mpcenv_b, ekf_b, plan_b)

    # EKF Matcher
    g0 = Observation()
    g1 = Observation()
    ekf_m = MatcherEKF(ekf_matcher_name, mpcenv_b, ekf_b, g0, g1)

    # ILQR Matcher
    ilqr_m = MatcherILQR(ilqr_matcher_name, ekf_b, plan_b, dynamics, cost, dt, T, iter_max)

    scheduler = VirtualTimeScheduler()

    # offset, interval, sleep
    timing_bundle = Timing(0, 1, 0)
    timing_matcher = Timing(1, 1, 0)
    timing_planning = Timing(1, control_T, 0)

    scheduler.add_component(mpcenv_b.component, timing_bundle)
    scheduler.add_component(ekf_b.component, timing_bundle)
    scheduler.add_component(ekf_m.component, timing_matcher)
    scheduler.add_component(controller_m.component, timing_matcher)
    scheduler.add_component(plan_b.component, timing_bundle)
    scheduler.add_component(ilqr_m.component, timing_planning)

    for i in range(num_steps):
        print("Step {}/{}".format(i, num_steps))
        scheduler.step()


if __name__ == '__main__':
    main()
