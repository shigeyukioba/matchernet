# -*- coding: utf-8 -*-
import autograd.numpy as np
import copy


class MultiAgentDynamics(object):
    """
    Dynamics wrapper for Multi Agent environment.
    """
    def __init__(self, dynamics, agent_size):
        self.dynamics = dynamics
        self.agent_size = agent_size

    def value(self, x, u):
        xdot_new = np.zeros_like(x)
        x_dim = self.dynamics.x_dim
        u_dim = self.dynamics.u_dim
        for i in range(self.agent_size):
            x_agent = x[x_dim*i:x_dim*(i+1)]
            u_agent = u[u_dim*i:u_dim*(i+1)]
            xdot_agent = self.dynamics.value(x_agent, u_agent)
            xdot_new[x_dim*i:x_dim*(i+1)] = xdot_agent
        return xdot_new

    def x(self, x, u):
        dx_new = np.zeros((self.x_dim, self.x_dim), dtype=np.float32)
        x_dim = self.dynamics.x_dim
        u_dim = self.dynamics.u_dim
        for i in range(self.agent_size):
            x_agent = x[x_dim*i:x_dim*(i+1)]
            u_agent = u[u_dim*i:u_dim*(i+1)]
            dx_agent = self.dynamics.x(x_agent, u_agent) # x_dim, x_dim
            dx_new[x_dim*i:x_dim*(i+1), x_dim*i:x_dim*(i+1)] = dx_agent
        return dx_new
        
    def u(self, x, u):
        du_new = np.zeros((self.x_dim, self.u_dim), dtype=np.float32)
        x_dim = self.dynamics.x_dim
        u_dim = self.dynamics.u_dim
        for i in range(self.agent_size):
            x_agent = x[x_dim*i:x_dim*(i+1)]
            u_agent = u[u_dim*i:u_dim*(i+1)]
            du_agent = self.dynamics.u(x_agent, u_agent) # (x_dim, u_dim)
            du_new[x_dim*i:x_dim*(i+1), u_dim*i:u_dim*(i+1)] = du_agent
        return du_new

    @property
    def x_dim(self):
        return self.dynamics.x_dim * self.agent_size

    @property
    def u_dim(self):
        return self.dynamics.u_dim * self.agent_size


class MultiAgentCost(object):
    def __init__(self, cost, agent_size):
        # TODO: costが状態を持つのをやめた方がいいだろう
        self.costs = [cost.clone() for i in range(agent_size)]
        self.agent_size = agent_size
        
    def clone(self):
        # TODO: costが状態を持つのをやめた方がいいだろう
        new_costs = [cost.clone() for cost in self.costs]
        new_multi_cost = copy.copy(self)
        new_multi_cost.costs = new_costs
        return new_multi_cost

    def apply_state(self, x, t):
        # TODO:
        x_dim = x.shape[0] // self.agent_size
        
        for i in range(self.agent_size):
            x_agent = x[x_dim*i:x_dim*(i+1)]
            self.costs[i].apply_state(x_agent, t)
    
    def value(self, x, u, t):
        x_dim = x.shape[0]
        u_dim = u.shape[0]
        
        value_new = 0.0
        for i in range(self.agent_size):
            x_agent = x[x_dim*i:x_dim*(i+1)]
            u_agent = u[u_dim*i:u_dim*(i+1)]
            # TODO:
            value_agent = self.costs[i].value(x_agent, u_agent, t)
            value_new += value_agent
        return value_new


class MultiAgentRenderer(object):
    """
    Renderer wrapper for Multi Agent environment.
    """
    def __init__(self, renderer, agent_size):
        self.renderer = renderer
        self.agent_size = agent_size

    def render(self, x, u=None):
        image = None
        for i in range(self.agent_size):
            image = self.renderer.render(x, u, override_image=image)
        return image


class MultiAgentRewardSystem(object):
    """
    RewardSystem wrapper for Multi Agent environment.
    """    
    def __init__(self, reward_system, agent_size):
        self.reward_system = reward_system
        self.agent_size = agent_size

    def reset(self):
        self.reward_system.reset()

    def render(self, image):
        return self.reward_system.render(image)

    def evaluate(self, x, dt):
        x_dim = x.shape[0] // self.agent_size        
        evaluated_reward = 0.0
        for i in range(self.agent_size):
            x_agent = x[x_dim*i:x_dim*(i+1)]
            evaluated_reward += self.reward_system.evaluate(x_agent, dt)
        return evaluated_reward
