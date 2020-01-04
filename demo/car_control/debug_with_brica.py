# -*- coding: utf-8 -*-
import numpy as np
import brica
from brica import Component, VirtualTimeScheduler, Timing

from matchernet import MPCEnv, MPCEnvBundle, MPCEnvDebugMatcher

from car import CarDynamics, CarRenderer
from obstacle_reward_system import ObstacleRewardSystem


def main():
    dt = 0.03
    dynamics = CarDynamics()
    renderer = CarRenderer()
    reward_system = ObstacleRewardSystem()
    env = MPCEnv(dynamics, renderer, reward_system, dt, use_visual_state=False)
    
    mpcenv_bundle = MPCEnvBundle(env)
    mpcenv_matcher = MPCEnvDebugMatcher(mpcenv_bundle)
    scheduler = VirtualTimeScheduler()
    
    timing0 = Timing(0, 1, 1)
    timing1 = Timing(1, 1, 1)
    
    scheduler.add_component(mpcenv_bundle.component, timing0)
    scheduler.add_component(mpcenv_matcher.component, timing1)

    num_steps = 10
    
    for i in range(num_steps):
        print("Step {}/{}".format(i, num_steps))
        scheduler.step()
    
if __name__ == '__main__':
    main()
