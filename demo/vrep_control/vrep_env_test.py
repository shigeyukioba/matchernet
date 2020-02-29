# -*- coding: utf-8 -*-
import numpy as np
import unittest

from vrep_env import VREPEnv


class VREPEnvTest(unittest.TestCase):
    def test_vrep_env(self):
        env = VREPEnv(direct=True)

        # x,u dimension check
        self.assertEqual(env.x_dim, 12)
        self.assertEqual(env.u_dim, 6)
        
        q = np.zeros(6, dtype=np.float32)

        env.reset_angles(q)
        env.set_target_shadow(q)

        u = np.zeros(6, dtype=np.float32)        

        state, reward = env.step(u)
        self.assertEqual(state.shape, (12,))

        env.close()
        
        
if __name__ == '__main__':
    unittest.main()
