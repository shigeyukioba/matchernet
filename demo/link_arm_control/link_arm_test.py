# -*- coding: utf-8 -*-
import numpy as np
import unittest

from link_arm import LinkArmDynamics


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


class LinkArmDynamicsTest(unittest.TestCase):
    def test_link_arm_dynamics(self):
        dt = 0.03
        
        dynamics = LinkArmDynamics(dt)
        
        x = np.zeros(4, dtype=np.float32)
        u = np.ones(2, dtype=np.float32)

        # Check shape of the next state
        x_next = dynamics.value(x, u)
        self.assertEqual(x_next.shape, (4,))

        # Check shape of the Jacobian w.r.t. x
        fx = dynamics.x(x, u)
        self.assertEqual(fx.shape, (4,4))

        # Compare Jacobian value with numerical differentiation result
        fx_n = jacobian_finite_difference(dynamics.value, 0, x, u)
        self.assertTrue(np.allclose(fx, fx_n, atol=1e-2))

        # Check shape of the Jacobian w.r.t. u
        fu = dynamics.u(x, u)
        self.assertEqual(fu.shape, (4,2))

        # Compare Jacobian value with numerical differentiation result
        fu_n = jacobian_finite_difference(dynamics.value, 1, x, u)
        self.assertTrue(np.allclose(fu, fu_n, atol=1e-3))


if __name__ == '__main__':
    unittest.main()
