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
             Renderer or list of Renderer (Agent renderer)
          reward_system:
             RewardSystem or None
          use_visual_state:
             Whether to use visual state output or not (bool)
        """
        self.dynamics = dynamics
        self.renderer = renderer
        self.reward_system = reward_system
        self.use_visual_state = use_visual_state
        self.reset()
        
    def reset(self, x_init=None):
        """
        Reset the environment:

        Arguments:
          x_init:
             Initial state (can be None)

        Returns:
          Current state after resetting
        """
        if x_init is None:
            self.x = np.zeros((self.dynamics.x_dim,), dtype=np.float32)
        else:
            self.x = np.copy(x_init)
        if self.reward_system is not None:
            self.reward_system.reset()
        return self.get_state(action=np.zeros((self.dynamics.u_dim,)))
    
    def get_state(self, action):
        if self.use_visual_state:
            return self.get_visual_state(action)
        else:
            return self.x
        
    def get_visual_state(self, action):
        if isinstance(self.renderer, list):
            # For multi angle view rendering
            renderer_size = len(self.renderer)
            #image = np.ones((renderer_size, IMAGE_WIDTH, IMAGE_WIDTH, 3), dtype=np.float32)
            images = []
            for i in range(renderer_size):
                # Render control object
                image = self.renderer[i].render(self.x, action)
                
                # Render rewards
                if self.reward_system is not None:
                    self.reward_system.render(image)
                images.append(image)
            return np.stack(images)
        else:
            # For single angle view rendering
            #image = np.ones((IMAGE_WIDTH, IMAGE_WIDTH, 3), dtype=np.float32)
            # Render control object
            image = self.renderer.render(self.x, action)
            
            # Render rewards
            if self.reward_system is not None:
                self.reward_system.render(image)
            
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
        if self.reward_system is not None:
            reward = self.reward_system.evaluate(self.x, self.dynamics.dt)
        else:
            reward = 0.0
        return self.get_state(action), reward
