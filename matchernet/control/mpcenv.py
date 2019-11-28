# -*- coding: utf-8 -*-
import numpy as np

IMAGE_WIDTH = 256


class MPCEnv(object):
    def __init__(self,
                 dynamics,
                 renderer,
                 reward_system,
                 use_visual_state=False):
        """
        Arguments:
          dynamics:
             Agent dynamics
          renderer:
             Agent renderer
          reward_system:
             RewardSystem
          use_visual_state:
             Whether to use visual state output or not (bool)
        """
        self.dynamics = dynamics
        self.renderer = renderer
        self.reward_system = reward_system
        self.use_visual_state = use_visual_state
        self.reset()
        
    def reset(self):
        """
        Reset the environment:
        
        Returns:
          Current state after resetting
        """
        self.x = np.zeros((self.dynamics.x_dim,), dtype=np.float32)
        self.reward_system.reset()
        return self.get_state(action=np.zeros((self.dynamics.u_dim,)))
    
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

    def step(self, action):
        """
        Step forward the environment.

        Arguments:
          action
            Control signal
        
        Returns:
          (state, reward)
        """
        self.x = self.dynamics.value(self.x, action)
        
        # Calculate reward
        reward = self.reward_system.evaluate(self.x, self.dynamics.dt)
        return self.get_state(action), reward
