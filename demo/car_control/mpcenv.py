# -*- coding: utf-8 -*-
import numpy as np

IMAGE_WIDTH = 256


class MPCEnv(object):
    def __init__(self, dynamics, renderer, reward_system, use_visual_state=False):
        self.dynamics = dynamics
        self.renderer = renderer
        self.reward_system = reward_system
        self.use_visual_state = use_visual_state
        
        self.reset()
        
    def reset(self):
        self.x = np.zeros((self.dynamics.x_dim,), dtype=np.float32)
        self.reward_system.reset()
        return self.get_state(action=[0,0])
    
    def get_state(self, action):
        if self.use_visual_state:
            return self.get_visual_state(action)
        else:
            return self.x
        
    def get_visual_state(self, action):
        image = np.ones((IMAGE_WIDTH, IMAGE_WIDTH, 3), dtype=np.float32)

        # Render rewards
        self.reward_system.render(image)

        # Render control object
        self.renderer.render(image, self.x, action)
        return image

    def debug_limit_agent_pos(self):
        # TODO: デバッグ用だが扱いを考える
        self.x[0] = np.clip(self.x[0], -1.0, 1.0)
        self.x[1] = np.clip(self.x[1], -1.0, 1.0)

    def step(self, action):
        self.x = self.dynamics.value(self.x, action)

        # Calculate reward
        reward = self.reward_system.evaluate(self.x, self.dynamics.dt)
        
        self.debug_limit_agent_pos()
        return self.get_state(action), reward
