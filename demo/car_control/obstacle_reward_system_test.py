# -*- coding: utf-8 -*-
import numpy as np
import unittest

from obstacle_reward_system import ObstacleRewardSystem


class ObstacleRewardSystemTest(unittest.TestCase):
    def test_obstacle_reward_system(self):
        reward_system = ObstacleRewardSystem()

        dt = 0.03
        x = np.zeros(4, dtype=np.float32)
        
        reward_system.reset()
        
        # Check calculated reward
        reward = reward_system.evaluate(x, dt)
        self.assertEqual(type(reward), float)

        image_width = 256
        image = np.ones((image_width, image_width, 3), dtype=np.float32)
        image = reward_system.render(image)

        # Check image shape, type and value range (0.0~1.0).
        self.assertEqual(image.shape, (256,256,3))
        self.assertEqual(image.dtype, np.float32)
        self.assertGreaterEqual(np.min(image), 0.0)
        self.assertLessEqual(np.max(image), 1.0)


if __name__ == '__main__':
    unittest.main()
