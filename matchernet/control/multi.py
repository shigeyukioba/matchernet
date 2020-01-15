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
        x_dim = x.shape[0] // self.agent_size
        if u is not None:
            u_dim = u.shape[0] // self.agent_size
        
        value_new = 0.0
        for i in range(self.agent_size):
            x_agent = x[x_dim*i:x_dim*(i+1)]
            if u is not None:
                u_agent = u[u_dim*i:u_dim*(i+1)]
            else:
                u_agent = None
            # TODO:
            value_agent = self.costs[i].value(x_agent, u_agent, t)
            value_new += value_agent
        return value_new

    def x(self, x, u, t):
        x_dim = x.shape[0] // self.agent_size
        if u is not None:
            u_dim = u.shape[0] // self.agent_size

        dx = np.zeros((x.shape[0],))
        for i in range(self.agent_size):
            x_agent = x[x_dim*i:x_dim*(i+1)]
            if u is not None:
                u_agent = u[u_dim*i:u_dim*(i+1)]
            else:
                u_agent = None
            dx_agent = self.costs[i].x(x_agent, u_agent, t)
            dx[x_dim*i:x_dim*(i+1)] = dx_agent
        return dx

    def u(self, x, u, t):
        x_dim = x.shape[0] // self.agent_size
        if u is not None:
            u_dim = u.shape[0] // self.agent_size

        du = np.zeros((u.shape[0],))
        for i in range(self.agent_size):
            x_agent = x[x_dim*i:x_dim*(i+1)]
            if u is not None:
                u_agent = u[u_dim*i:u_dim*(i+1)]
            else:
                u_agent = None
            du_agent = self.costs[i].u(x_agent, u_agent, t)
            du[u_dim*i:u_dim*(i+1)] = du_agent
        return du

    def xx(self, x, u, t):
        x_dim = x.shape[0] // self.agent_size
        if u is not None:
            u_dim = u.shape[0] // self.agent_size

        dxx = np.zeros((x.shape[0],x.shape[0]))
        for i in range(self.agent_size):
            x_agent = x[x_dim*i:x_dim*(i+1)]
            if u is not None:
                u_agent = u[u_dim*i:u_dim*(i+1)]
            else:
                u_agent = None
            dxx_agent = self.costs[i].xx(x_agent, u_agent, t)
            dxx[x_dim*i:x_dim*(i+1),x_dim*i:x_dim*(i+1)] = dxx_agent
        return dxx

    def uu(self, x, u, t):
        x_dim = x.shape[0] // self.agent_size
        if u is not None:
            u_dim = u.shape[0] // self.agent_size

        duu = np.zeros((u.shape[0],u.shape[0]))
        for i in range(self.agent_size):
            x_agent = x[x_dim*i:x_dim*(i+1)]
            if u is not None:
                u_agent = u[u_dim*i:u_dim*(i+1)]
            else:
                u_agent = None
            duu_agent = self.costs[i].uu(x_agent, u_agent, t)
            duu[u_dim*i:u_dim*(i+1),u_dim*i:u_dim*(i+1)] = duu_agent
        return duu

    def ux(self, x, u, t):
        x_dim = x.shape[0] // self.agent_size
        if u is not None:
            u_dim = u.shape[0] // self.agent_size

        dux = np.zeros((u.shape[0],x.shape[0]))
        for i in range(self.agent_size):
            x_agent = x[x_dim*i:x_dim*(i+1)]
            if u is not None:
                u_agent = u[u_dim*i:u_dim*(i+1)]
            else:
                u_agent = None
            dux_agent = self.costs[i].ux(x_agent, u_agent, t)
            dux[u_dim*i:u_dim*(i+1),x_dim*i:x_dim*(i+1)] = dux_agent
        return dux



class MultiAgentRenderer(object):
    """
    Renderer wrapper for Multi Agent environment.
    """
    def __init__(self, renderer, agent_size):
        self.renderer = renderer
        self.agent_size = agent_size

    def render(self, x, u=None):
        x_dim = x.shape[0] // self.agent_size
        if u is not None:
            u_dim = u.shape[0] // self.agent_size
        
        image = None
        for i in range(self.agent_size):
            x_agent = x[x_dim*i:x_dim*(i+1)]
            if u is not None:
                u_agent = u[u_dim*i:u_dim*(i+1)]
            else:
                u_agent = None            
            image = self.renderer.render(x_agent, u_agent, override_image=image)
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
