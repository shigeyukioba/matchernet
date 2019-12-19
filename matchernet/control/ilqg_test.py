# -*- coding: utf-8 -*-
import autograd.numpy as np
import unittest
from autograd import jacobian, grad

from mpc import Dynamics
from ilqg import iLQG


class DummyDynamics(Dynamics):
    def __init__(self, dt):
        super(DummyDynamics, self).__init__(dt)
        
        self.x = jacobian(self.value, 0)
        self.u = jacobian(self.value, 1)

    def value(self, x, u):
        return x

    @property
    def x_dim(self):
        return 4

    @property
    def u_dim(self):
        return 2


class DummyCost(object):
    def __init__(self):
        self.x  = grad(self.value, 0)
        self.u  = grad(self.value, 1)
        self.xx = jacobian(self.x, 0)
        self.uu = jacobian(self.u, 1)
        self.ux = jacobian(self.u, 0)

    def clone(self):
        return self

    def apply_state(self, x, t):
        pass

    def value(self, x, u, t):
        taret_x = np.array([1,1,2,3], dtype=np.float32)
        x_cost = 1.0 * (x - taret_x).T @ (x - taret_x)
        
        if u is not None:
            # Running cost
            u_cost = 0.1 * (u[0]**2) + 0.2 * (u[1]**2)
            return x_cost + u_cost
        else:
            # Terminal cost
            return x_cost


class iLQGTest(unittest.TestCase):
    def test_ilqg(self):
        dt = 0.03
        dynamics = DummyDynamics(dt)
        cost = DummyCost()
        ilqg = iLQG(dynamics, cost)

        # Initial state
        x0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        T = 20
        receding_T = 10
        iter_max = 20

        # Initial control sequence
        u0 = np.zeros((T, 2), dtype=np.float32)

        x_list, u_list, K_list = ilqg.optimize(x0,
                                               u0,
                                               T,
                                               start_time_step=0,
                                               iter_max=iter_max)
        
        self.assertEqual(x_list.shape, (T+1,4))
        self.assertEqual(u_list.shape, (T,2))
        self.assertEqual(K_list.shape, (T,2,4))


if __name__ == '__main__':
    unittest.main()
