# -*- coding: utf-8 -*-
import autograd.numpy as np
import unittest
from autograd import grad, jacobian

from multi import MultiAgentDynamics, MultiAgentCost, MultiAgentRenderer, MultiAgentRewardSystem
from mpc import Dynamics, Cost, Renderer


class DummyDynamics(Dynamics):
    def __init__(self):
        super(DummyDynamics, self).__init__()
        
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


class DummyCost(Cost):
    def __init__(self):
        super(DummyCost, self).__init__()

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
        #value = np.sum(x @ x.T) + np.sum(u @ u.T)
        value = np.sum(x) + np.sum(u)
        return value


class DummyRenderer(Renderer):
    def __init__(self):
        super(DummyRenderer, self).__init__()

    def render(self, x, u, override_image=None):
        if override_image is not None:
            return override_image
        else:
            return np.ones([256, 256, 3], dtype=np.float32)


class DummyRewardSystem(object):
    def __init__(self):
        pass

    def reset(self):
        pass

    def render(self, image):
        return image

    def evaluate(self, x, dt):
        return 1.0


class MutlAgentTest(unittest.TestCase):
    def test_multi_agent_dynamics(self):
        dynamcis = DummyDynamics()
        multi_dynamics = MultiAgentDynamics(dynamcis, 10)

        x = np.zeros(4*10, dtype=np.float32)
        u = np.ones(2*10, dtype=np.float32)
        
        xdot = multi_dynamics.value(x, u)
        self.assertEqual(xdot.shape, (4*10,))

        dx = multi_dynamics.x(x, u)
        self.assertEqual(dx.shape, (4*10,4*10))

        du = multi_dynamics.u(x, u)
        self.assertEqual(du.shape, (4*10,2*10))

        self.assertEqual(multi_dynamics.x_dim, 4*10)
        self.assertEqual(multi_dynamics.u_dim, 2*10)

    def test_multi_agent_cost(self):
        cost = DummyCost()
        multi_cost = MultiAgentCost(cost, 10)

        x = np.zeros(4*10, dtype=np.float32)
        u = np.ones(2*10, dtype=np.float32)

        t = 0.0
        v = multi_cost.value(x, u, t)
        #self.assertEqual(type(v), float)
        
        multi_cost.clone()
        multi_cost.apply_state(x, t)

        dx = multi_cost.x(x, u, t)
        self.assertEqual(dx.shape, (40,))
        du = multi_cost.u(x, u, t)
        self.assertEqual(du.shape, (20,))
        dxx = multi_cost.xx(x, u, t)
        self.assertEqual(dxx.shape, (40,40))
        duu = multi_cost.uu(x, u, t)
        self.assertEqual(duu.shape, (20,20))
        dux = multi_cost.ux(x, u, t)
        self.assertEqual(dux.shape, (20,40))

    def test_multi_agent_renderer(self):
        renderer = DummyRenderer()
        multi_renderer = MultiAgentRenderer(renderer, 10)

        x = np.zeros(4*10, dtype=np.float32)
        u = np.ones(2*10, dtype=np.float32)

        image = multi_renderer.render(x, u)
        self.assertEqual(image.shape, (256,256,3))

    def test_multi_agent_rewared_system(self):
        reward_system = DummyRewardSystem()
        multi_reward_sysetm = MultiAgentRewardSystem(reward_system, 10)
        
        x = np.zeros(4*10, dtype=np.float32)
        image = np.ones([256, 256, 3], dtype=np.float32)
        dt = 0.1

        multi_reward_sysetm.reset()
        image = multi_reward_sysetm.render(image)
        self.assertEqual(image.shape, (256,256,3))

        evaluated_reward = multi_reward_sysetm.evaluate(x, dt)
        self.assertEqual(type(evaluated_reward), float)


if __name__ == '__main__':
    unittest.main()
