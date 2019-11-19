# -*- coding: utf-8 -*-
from matchernet import Bundle, Matcher
from mpcenv import MPCEnv
from car import CarDynamics, CarRenderer
from obstacle_reward_system import ObstacleRewardSystem


class MPCEnvBundle(Bundle):
    def __init__(self):
        super(MPCEnvBundle, self).__init__("mpcenv_b0")

        dt = 0.03
        dynamics = CarDynamics(dt)
        renderer = CarRenderer()
        reward_system = ObstacleRewardSystem()
        
        self.env = MPCEnv(dynamics, renderer, reward_system, use_visual_state=False)
        self.update_component()

        self.timestamp = 0.0

    def __call__(self, inputs):
        action = inputs["u"]
        state, reward = self.env.step(action)
        
        # Increment timestamp
        self.timestamp += self.env.dynamics.dt
        
        return {
            "x" : state,
            "r" : reward,
            "timestamp" : self.timestamp
        }
