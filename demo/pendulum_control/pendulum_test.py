# -*- coding: utf-8 -*-
import numpy as np
import unittest

from pendulum import PendulumDynamics, PendulumCost, PendulumRenderer

DEBUG_SAVE_STATE = False

if DEBUG_SAVE_STATE:
    from scipy.misc import imsave


# TODO unity same func in car_test.py
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


class PendulumDynamicsTest(unittest.TestCase):
    def test_pendulum_dynamics(self):
        dt = 0.05
        dynamics = PendulumDynamics(dt)
        
        x = np.zeros(2, dtype=np.float32)
        u = np.ones(1, dtype=np.float32)

        # Check shape of the next state
        x_next = dynamics.value(x, u)
        self.assertEqual(x_next.shape, (2,))

        # Check shape of the Jacobian w.r.t. x
        fx = dynamics.x(x, u)
        self.assertEqual(fx.shape, (2,2))

        # Compare Jacobian value with numerical differentiation result
        fx_n = jacobian_finite_difference(dynamics.value, 0, x, u)
        self.assertTrue(np.allclose(fx, fx_n, atol=1e-3))

        # Check shape of the Jacobian w.r.t. u
        fu = dynamics.u(x, u)
        self.assertEqual(fu.shape, (2,1))

        # Compare Jacobian value with numerical differentiation result
        fu_n = jacobian_finite_difference(dynamics.value, 1, x, u)
        np.testing.assert_almost_equal(fu, fu_n, 2)
        self.assertTrue(np.allclose(fu, fu_n, atol=1e-3))


class PendulumCostTest(unittest.TestCase):
    def test_pendulum_cost(self):
        cost = PendulumCost()
        
        x = np.zeros(2, dtype=np.float32)
        u = np.ones(1, dtype=np.float32)

        # TODO: check with numerical differentiation
        c = cost.value(x, u, 0)
        self.assertEqual(c.shape, ())
        
        cx = cost.x(x, u, 0)
        self.assertEqual(cx.shape, (2,))
        
        cu = cost.u(x, u, 0)
        self.assertEqual(cu.shape, (1,))
        
        cxx = cost.xx(x, u, 0)
        self.assertEqual(cxx.shape, (2,2))
        
        cuu = cost.uu(x, u, 0)
        self.assertEqual(cuu.shape, (1,1))
        
        cux = cost.ux(x, u, 0)
        self.assertEqual(cux.shape, (1,2))

        
class PendulumRendererTest(unittest.TestCase):
    def test_pendulum_renderer(self):
        renderer = PendulumRenderer()
        
        x = np.zeros(2, dtype=np.float32)
        u = np.ones(1, dtype=np.float32)

        image = np.ones((256, 256, 3), dtype=np.float32)
        renderer.render(image, x, u)

        self.assertEqual(image.shape, (256,256,3))

        if DEBUG_SAVE_STATE:
            imsave("pendulum_test.png", image)



if __name__ == '__main__':
    unittest.main()
