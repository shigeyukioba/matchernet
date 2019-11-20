# -*- coding: utf-8 -*-
import numpy as np
import cv2

from matchernet import RewardSystem


class Reward(object):
    def __init__(self, pos, is_good, radius=0.1):
        self.pos = pos
        self.is_good = is_good
        self.radius = radius

    def contains(self, agent_pos):
        d = agent_pos - self.pos
        return np.sum(d**2) < self.radius**2

    def render(self, image):
        image_width = image.shape[1]
        render_scale = image_width / 2.0
        render_offset = image_width / 2.0

        rx = int(self.pos[0] * render_scale + render_offset)
        ry = int(self.pos[1] * render_scale + render_offset)
        if self.is_good:
            reward_color = (1,0,0)
        else:
            reward_color = (0,0,0)
        image = cv2.circle(image,
                           (rx, ry), int(self.radius * render_scale),
                           reward_color, -1)
        return image

    def evaluate(self, x, dt):
        pos = x[:2]
        if self.contains(pos):
            if self.is_good:
                return 1.0 * dt
            else:
                return -1.0 * dt
        else:
            return 0.0


class ObstacleRewardSystem(RewardSystem):
    def __init__(self):
        super(ObstacleRewardSystem, self).__init__()
        self.rewards = []

    def reset(self):
        self.locate_rewards()
        
    def locate_rewards(self):
        self.rewards = []

        # Locate rewards to random locations
        x0 = np.random.uniform(-0.8, 0.8, 1)
        y0 = np.random.uniform(-0.8, 0.8, 1)
        x1 = np.random.uniform(-0.8, 0.8, 1)
        y1 = np.random.uniform(-0.8, 0.8, 1)
        
        good_reward = Reward(pos=np.array((x0, y0), dtype=np.float32),
                             is_good=True)
        bad_reward = Reward(pos=np.array((x1, y1), dtype=np.float32),
                            is_good=False)
        self.rewards.append(good_reward)
        self.rewards.append(bad_reward)

    def render(self, image):
        """ Render rewards """
        for reward in self.rewards:
            reward.render(image)
        return image

    def evaluate(self, x, dt):
        """ Evaluate reward based on current state """
        evaluated_reward = 0.0
        for reward in self.rewards:
            evaluated_reward += reward.evaluate(x, dt)
        return evaluated_reward
