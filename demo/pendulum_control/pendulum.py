# -*- coding: utf-8 -*-
import autograd.numpy as np
from autograd import jacobian, grad

from matchernet import Dynamics, Renderer, Cost

import cv2


class PendulumDynamics(Dynamics):
    """
    Inverted pendulum dynamics.
    """
    def __init__(self):
        super(PendulumDynamics, self).__init__()

        # Constants
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        
        # Jacobian calculation with automatic differentiation
        self.x = jacobian(self.value, 0)
        self.u = jacobian(self.value, 1)

    def value(self, x, u):
        th    = x[0]
        thdot = x[1]
        thdotdot = (-3 * self.g / (2 * self.l) * np.sin(th + np.pi) + 3.0 / (self.m * self.l ** 2) * u[0])
        return np.array([thdot, thdotdot], dtype=np.float32)

    @property
    def x_dim(self):
        return 2

    @property
    def u_dim(self):
        return 1


class PendulumCost(Cost):
    """
    Inverted pendulum cost function.
    """
    def __init__(self):
        super(PendulumCost, self).__init__()
        
        self.x  = grad(self.value, 0)
        self.u  = grad(self.value, 1)
        self.xx = jacobian(self.x, 0)
        self.uu = jacobian(self.u, 1)
        self.ux = jacobian(self.u, 0)

    def value(self, x, u, t):
        """ Calculate cost.
        If u is None, cost is calculated as terminal cost. If u is not None,
        then cost is calculated as running cost.
        """
        def angle_normalize(x):
            return (((x+np.pi) % (2*np.pi)) - np.pi)
        
        th    = x[0]
        thdot = x[1]
        if u is not None:
            # Runnning cost
            cost = angle_normalize(th)**2 + 0.1 * thdot**2 + 0.001 * (u[0]**2)
        else:
            # Terminal cost
            cost = angle_normalize(th)**2 + 0.1 * thdot**2
        return cost



ROD_LENGTH = 0.5
ROD_WIDTH  = 0.05


class PendulumRenderer(Renderer):
    """
    Inverted pendulum renderer.
    """
    def __init__(self, image_width=256):
        super(PendulumRenderer, self).__init__()
        self.image_width = image_width

    def draw_rotated_rect(self, image, x, y, w, h, angle, color):
        rect = ((x,y), (h, w), angle / np.pi * 180.0)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        image = cv2.drawContours(image, [box], 0, color, -1)
        return image

    def render(self, x, u):
        image = np.ones((self.image_width, self.image_width, 3), dtype=np.float32)
        
        render_scale = self.image_width / 2.0
        render_offset = self.image_width / 2.0

        # TODO: angleを修正
        angle = x[0] - np.pi * 0.5
        
        cener_x = np.cos(angle) * ROD_LENGTH/2
        cener_y = np.sin(angle) * ROD_LENGTH/2

        image = self.draw_rotated_rect(image,
                                       cener_x * render_scale + render_offset,
                                       cener_y * render_scale + render_offset,
                                       ROD_WIDTH * render_scale,
                                       ROD_LENGTH * render_scale,
                                       angle,
                                       (0,0,0))
        image = cv2.circle(image,
                           (int(render_offset), int(render_offset)),
                           int(0.04 * render_scale),
                           (1,0,0), -1)
        return image
