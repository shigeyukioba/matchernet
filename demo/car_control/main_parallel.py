# -*- coding: utf-8 -*-
import numpy as np
from autograd import jacobian
import cv2

import brica
from brica import VirtualTimeScheduler, Timing

from matchernet import MPCEnvBundle, MatcherController, Plan, MatcherILQR, BundlePlan, BundleEKFWithController, MatcherEKF, MPCEnv, MovieWriter, AnimGIFWriter

from car import CarDynamics, CarCost, CarRenderer, CarObstacle


class CarObservation(object):
    def __init__(self):
        # Jacobian calculation with automatic differentiation
        self.x = jacobian(self.value, 0)
        
    def value(self, x):
        return x


class CarEnvRecorder(object):
    def __init__(self, recording_frame_size, cost):
        self.recording_frame_size = recording_frame_size
        self.current_frame = 0
        self.obstacles = cost.obstacles
        
    def record(self, x, u):
        if self.current_frame == 0:
            self.renderer = CarRenderer(image_width=256)
            self.movie = MovieWriter("out.mov", (256, 256), 30)
            self.gif = AnimGIFWriter("out.gif", 30)
        
        if self.current_frame < self.recording_frame_size:
            image = self.renderer.render(x, u)

            self.render_obstacles(image, self.obstacles)
            
            image = (image * 255.0).astype(np.uint8)
            self.movie.add_frame(image)
            self.gif.add_frame(image)
            if self.current_frame == self.recording_frame_size-1:
                self.movie.close()
                self.gif.close()
        self.current_frame += 1

    def render_obstacles(self, image, obstacles):
        image_width = image.shape[1]
        render_scale = image_width / 2.0
        render_offset = image_width / 2.0
    
        for obstacle in obstacles:
            rx = int(obstacle.pos[0] * render_scale + render_offset)
            ry = int(obstacle.pos[1] * render_scale + render_offset)
            rate = 1.0 # TODO: Rewardの消える反映が未対応
            if rate > 1.0:
                rate = 1.0
            if obstacle.is_good:
                reward_color = (1, 1-rate, 1-rate)
            else:
                reward_color = (1-rate, 1-rate, 1-rate)
            radius = 0.1
            image = cv2.circle(image,
                               (rx, ry), int(radius * render_scale),
                               reward_color, -1)

        
def prepare_cost():
    obstacles = []
    
    obstacle0 = CarObstacle(pos=np.array([0.5, 0.0], dtype=np.float32),
                            is_good=False)
    obstacles.append(obstacle0)
    obstacle1 = CarObstacle(pos=np.array([0.5, 0.3], dtype=np.float32),
                            is_good=True)
    obstacles.append(obstacle1)
    """
    obstacle2 = CarObstacle(pos=np.array([0.8, 0.6], dtype=np.float32),
                            is_good=False)
    obstacles.append(obstacle2)
    obstacle3 = CarObstacle(pos=np.array([0.6, 0.8], dtype=np.float32),
                            is_good=True)
    obstacles.append(obstacle3)
    """
    
    cost = CarCost(obstacles)
    return cost


def main():
    np.random.rand(0)

    dt = 0.03
    dynamics = CarDynamics()
    cost = prepare_cost()
    T = 50 # MPC Horizon
    control_T = 10 # Plan update interval for receding horizon
    iter_max = 10
    num_steps = 400

    # Component names
    ekf_controller_bundle_name = "ekf_contrller_bundle"
    ekf_matcher_name = "ekf_matcher"
    plan_bundle_name = "plan_bundle"
    controller_matcher_name = "controller_matcher"
    ilqr_matcher_name = "ilqr_matcher"
    mpcenv_bundle_name = "mpc_env_bundle"

    # Initial state
    x0 = np.zeros((4,), dtype=np.float32)

    # Initial internal state
    mu0 = np.zeros((4,), dtype=np.float32)
    Sigma0 = np.eye(4, dtype=np.float32) * 0.001
    
    # System noise covariance
    Q = np.eye(4, dtype=np.float32) * 0.0001

    # Observation noise
    R = np.eye(4, dtype=np.float32) * 0.0001
    
    # MPCEnv Bundle
    env = MPCEnv(dynamics, None, None, dt, use_visual_state=False)
    env.reset(x0)
    debug_recorder = CarEnvRecorder(num_steps//2-1, cost)
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
    g0 = CarObservation()
    g1 = CarObservation()
    ekf_m = MatcherEKF(ekf_matcher_name, mpcenv_b, ekf_b, g0, g1)

    # ILQR Matcher
    ilqr_m = MatcherILQR(ilqr_matcher_name, ekf_b, plan_b, dynamics, cost, dt, T, iter_max)

    scheduler = VirtualTimeScheduler()

    # offset, interval, sleep
    timing_bundle = Timing(0, 1, 1)
    timing_matcher = Timing(1, 1, 1)
    timing_planning = Timing(1, control_T*2, 0)

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
