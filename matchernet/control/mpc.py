# -*- coding: utf-8 -*-

class Dynamics(object):
    """
    Base class for MPC dynamics definition.
    """
    def __init__(self, dt):
        # Timestep
        self._dt = dt

    def value(self, x, u):
        raise NotImplementedError()

    def x(self, x, u):
        raise NotImplementedError()

    def u(self, x, u):
        raise NotImplementedError()

    @property
    def x_dim(self):
        raise NotImplementedError()

    @property
    def u_dim(self):
        raise NotImplementedError()

    @property
    def dt(self):
        return self._dt


class Renderer(object):
    """
    Base class for MPC agent renderer.
    """    
    def __init__(self):
        pass

    def render(self, image, x, u=None):
        raise NotImplementedError()


class RewardSystem(object):
    """
    Base reward system class for MPC environment.
    """
    def __init__(self):
        pass

    def render(self, image):
        raise NotImplementedError()

    def evaluate(self, x, dt):
        raise NotImplementedError()
