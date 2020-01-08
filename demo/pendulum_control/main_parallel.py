# -*- coding: utf-8 -*-
import numpy as np
from autograd import jacobian

import brica
from brica import VirtualTimeScheduler, Timing

from matchernet import MPCEnvBundle, MatcherController, Plan, MatcherILQR, BundlePlan, BundleEKFWithController, MatcherEKF, MPCEnv, MovieWriter, AnimGIFWriter

from pendulum import PendulumDynamics, PendulumCost, PendulumRenderer


class PendulumObservation(object):
    def __init__(self):
        # Jacobian calculation with automatic differentiation
        self.x = jacobian(self.value, 0)
        
    def value(self, x):
        return x


class PendulumEnvRecorder(object):
    def __init__(self, recording_frame_size):
        self.recording_frame_size = recording_frame_size
        self.current_frame = 0
        
    def record(self, x, u):
        if self.current_frame == 0:
            self.renderer = PendulumRenderer(image_width=256)
            self.movie = MovieWriter("out.mov", (256, 256), 30)
            self.gif = AnimGIFWriter("out.gif", 30)
        
        if self.current_frame < self.recording_frame_size:
            image = self.renderer.render(x, u)
            image = (image * 255.0).astype(np.uint8)
            self.movie.add_frame(image)
            self.gif.add_frame(image)
            if self.current_frame == self.recording_frame_size-1:
                self.movie.close()
                self.gif.close()
        self.current_frame += 1
        

def main():
    np.random.rand(0)

    dt = 0.02
    #dt = 0.05
    dynamics = PendulumDynamics()
    cost = PendulumCost()
    #T = 30 # MPC Horizon
    T = 40 # MPC Horizon
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

    # Initial state
    x0 = np.array([np.pi, 0.0], dtype=np.float32)

    # Initial internal state
    mu0 = np.array([np.pi, 0.0], dtype=np.float32)
    Sigma0 = np.array([[0.001, 0.0], [0.0, 0.001]], dtype=np.float32)

    # System noise covariance
    Q = np.array([[0.01, 0.0], [0.0, 0.01]], dtype=np.float32)

    # Observation noise
    R = np.array([[0.01, 0.0], [0.0, 0.01]], dtype=np.float32)
    
    # MPCEnv Bundle
    env = MPCEnv(dynamics, None, None, dt, use_visual_state=False)
    env.reset(x0)
    debug_recorder = PendulumEnvRecorder(num_steps-5)
    mpcenv_b = MPCEnvBundle(mpcenv_bundle_name, env, R,
                            controller_matcher_name,
                            debug_recorder=debug_recorder)

    # EKF Controller Bundle
    ekf_b = BundleEKFWithController(ekf_controller_bundle_name, dt, dynamics, Q,
                                    mu0, Sigma0, controller_matcher_name)
    
    # Plan Bundle
    plan_b = BundlePlan(plan_bundle_name, dynamics.x_dim, dynamics.u_dim, dt, control_T,
                        ilqr_matcher_name)

    # Controller Matcher
    controller_m = MatcherController(controller_matcher_name, mpcenv_b, ekf_b, plan_b)

    # EKF Matcher
    g0 = PendulumObservation()
    g1 = PendulumObservation()
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
