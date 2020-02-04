# -*- coding: utf-8 -*-

class Dynamics(object):
    """
    Base interfaces for MPC dynamics definition.
    """
    def __init__(self):
        pass

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


class Renderer(object):
    """
    Base interfaces for MPC agent renderer.
    """
    def __init__(self):
        pass

    def render(self, x, u=None, override_image=None):
        raise NotImplementedError()


class RewardSystem(object):
    """
    Base interfaces system class for MPC environment.
    """
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError()

    def render(self, image):
        raise NotImplementedError()

    def evaluate(self, x, dt):
        raise NotImplementedError()


class Cost(object):
    def __init__(self):
        pass

    def clone(self):
        return self

    def apply_state(self, x, t):
        pass

    def value(self, x, u, t):
        raise NotImplementedError()
