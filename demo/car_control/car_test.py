# -*- coding: utf-8 -*-
import numpy as np
import unittest

from car import CarDynamics, CarCost, CarObstacle
from matchernet import MultiAgentDynamics, MultiAgentCost, MultiAgentRenderer, MultiAgentRewardSystem


def jacobian_finite_difference(func, arg_index, *args):
    """ Calculate Jacobian with finite differences. """ 
    eps = 1e-5

    dim_out = func(*args).shape[0]
    dim_in = args[arg_index].shape[0]
    J = np.zeros([dim_out, dim_in], dtype=np.float32)

    for i in range(dim_in):
        args0 = [arg.copy() for arg in args]
        args1 = [arg.copy() for arg in args]
        
        args0[arg_index][i] += eps
        args1[arg_index][i] -= eps
        
        f0 = func(*args0)
        f1 = func(*args1)
        
        J[:,i] = (f0-f1) / (2*eps)
    return J


class CarDynamicsTest(unittest.TestCase):
    def test_car_dynamics(self):
        dynamics = CarDynamics()
        
        x = np.zeros(4, dtype=np.float32)
        u = np.ones(2, dtype=np.float32)

        # Check shape of the next state
        dx = dynamics.value(x, u)
        self.assertEqual(dx.shape, (4,))

        # Check shape of the Jacobian w.r.t. x
        fx = dynamics.x(x, u)
        self.assertEqual(fx.shape, (4,4))

        # Compare Jacobian value with numerical differentiation result
        fx_n = jacobian_finite_difference(dynamics.value, 0, x, u)
        self.assertTrue(np.allclose(fx, fx_n, atol=1e-4))

        # Check shape of the Jacobian w.r.t. u
        fu = dynamics.u(x, u)
        self.assertEqual(fu.shape, (4,2))
        
        # Compare Jacobian value with numerical differentiation result
        fu_n = jacobian_finite_difference(dynamics.value, 1, x, u)
        self.assertTrue(np.allclose(fu, fu_n, atol=1e-2))


class CarCostTest(unittest.TestCase):
    def test_car_cost(self):
        obstacles = []
        obstacle0 = CarObstacle(pos=np.array([0.5, 0.0], dtype=np.float32),
                                is_good=False)
        obstacles.append(obstacle0)
        obstacle1 = CarObstacle(pos=np.array([0.5, 0.3], dtype=np.float32),
                                is_good=True)
        obstacles.append(obstacle1)
        
        cost = CarCost(obstacles)

        x = np.zeros((4,), dtype=np.float32)
        u = np.zeros((2,), dtype=np.float32)

        t = 0
        
        l_terminal = cost.value(x, None, t)
        self.assertEqual(l_terminal.shape, ())

        l = cost.value(x, u, t)
        self.assertEqual(l.shape, ())
        
        lx = cost.x(x, u, t)
        self.assertEqual(lx.shape, (4,))
        
        lu = cost.u(x, u, t)
        self.assertEqual(lu.shape, (2,))

        cloned_cost = cost.clone()


class MultiAgentTest(unittest.TestCase):
    def test_multi_agent(self):
        agent_size = 2

        x = np.zeros(4*agent_size, dtype=np.float32)
        u = np.ones(2*agent_size, dtype=np.float32)        

        # Check multi agent dynamics
        dynamics = MultiAgentDynamics(CarDynamics(), agent_size)
        xdot = dynamics.value(x, u)
        self.assertEqual(xdot.shape, (4*agent_size,))

        dx = dynamics.x(x, u)
        self.assertEqual(dx.shape, (4*agent_size,4*agent_size))

        du = dynamics.u(x, u)
        self.assertEqual(du.shape, (4*agent_size,2*agent_size))

        self.assertEqual(dynamics.x_dim, 4*agent_size)
        self.assertEqual(dynamics.u_dim, 2*agent_size)

        # Check multi agent cost
        obstacles = []
        obstacle0 = CarObstacle(pos=np.array([0.5, 0.0], dtype=np.float32), is_good=False)
        obstacles.append(obstacle0)
        obstacle1 = CarObstacle(pos=np.array([0.5, 0.3], dtype=np.float32), is_good=True)
        obstacles.append(obstacle1)
        
        cost = MultiAgentCost(CarCost(obstacles), agent_size)
        t = 0.0

        v = cost.value(x, u, t)
        #self.assertEqual(type(v), float)
        
        cost.clone()
        cost.apply_state(x, t)

        dx = cost.x(x, u, t)
        self.assertEqual(dx.shape, (4*agent_size,))
        du = cost.u(x, u, t)
        self.assertEqual(du.shape, (2*agent_size,))
        dxx = cost.xx(x, u, t)
        self.assertEqual(dxx.shape, (4*agent_size,4*agent_size))
        duu = cost.uu(x, u, t)
        self.assertEqual(duu.shape, (2*agent_size,2*agent_size))
        dux = cost.ux(x, u, t)
        self.assertEqual(dux.shape, (2*agent_size,4*agent_size))
        
        
if __name__ == '__main__':
    unittest.main()
