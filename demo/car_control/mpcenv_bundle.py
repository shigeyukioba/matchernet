# -*- coding: utf-8 -*-
from matchernet import Bundle, Matcher
from mpcenv import MPCEnv
from car import CarDynamics, CarRenderer
from obstacle_reward_system import ObstacleRewardSystem


class MPCEnvBundle(Bundle):
    def __init__(self):
        super(MPCEnvBundle, self).__init__("mpcenv")

        dt = 0.03
        dynamics = CarDynamics(dt)
        renderer = CarRenderer()
        reward_system = ObstacleRewardSystem()
        
        self.env = MPCEnv(dynamics, renderer, reward_system, use_visual_state=False)
        self.update_component()

    def __call__(self, inputs):
        action = [0.0, 0.1]
        new_state, reward = self.env.step(action)
        print("new state={}".format(new_state))
        return {}
