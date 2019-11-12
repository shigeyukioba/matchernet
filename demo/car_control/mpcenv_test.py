# -*- coding: utf-8 -*-
import numpy as np
import unittest

from car import CarDynamics, CarRenderer
from obstacle_reward_system import ObstacleRewardSystem
from mpcenv import MPCEnv


class MPCEnvTest(unittest.TestCase):
    def test_mpcenv_without_visual(self):
        """
        Test MPCEnv with normal vector state output.
        """
        dt = 0.03
        dynamics = CarDynamics(dt)
        renderer = CarRenderer()
        reward_system = ObstacleRewardSystem()

        # Create environment with vector state output.
        env = MPCEnv(dynamics, renderer, reward_system, use_visual_state=False)

        state = env.reset()
        # Check state shape and type.
        self.assertEqual(state.shape, (4,))
        self.assertEqual(state.dtype, np.float32)

        for i in range(10):
            action = [0.1, 1.0]
            state, reward = env.step(action)

            # Check state shape and type.
            self.assertEqual(state.shape, (4,))
            self.assertEqual(state.dtype, np.float32)

            # Check reward is a scalar value
            self.assertEqual(type(reward), float)

    def test_mpcenv_with_visual(self):
        """
        Test MPCEnv with image state output.
        """
        dt = 0.03
        dynamics = CarDynamics(dt)
        renderer = CarRenderer()
        reward_system = ObstacleRewardSystem()

        # Create environment with image state output.
        env = MPCEnv(dynamics, renderer, reward_system, use_visual_state=True)

        state = env.reset()
        
        # Check state shape, type and value range (0.0~1.0).
        self.assertEqual(state.shape, (256,256,3))
        self.assertEqual(state.dtype, np.float32)
        self.assertGreaterEqual(np.min(state), 0.0)
        self.assertLessEqual(np.max(state), 1.0)

        for i in range(10):
            action = [0.1, 1.0]
            state, reward = env.step(action)

            # Check state shape, type and value range (0.0~1.0).
            self.assertEqual(state.shape, (256,256,3))
            self.assertEqual(state.dtype, np.float32)
            self.assertGreaterEqual(np.min(state), 0.0)
            self.assertLessEqual(np.max(state), 1.0)

            # Check reward is a scalar value
            self.assertEqual(type(reward), float)


if __name__ == '__main__':
    unittest.main()
    
