# -*- coding: utf-8 -*-
import autograd.numpy as np
from autograd import jacobian, grad
import cv2
import copy

from matchernet import Dynamics, Renderer, Cost


# Dynamics properties
CAR_AXLES_DISTANCE = 2.0 * 0.1 # Distance between axles

# Rendering properties
CAR_WIDTH          = 0.9 * 0.1
CAR_LENGTH         = 2.1 * 0.1
WHEEL_WIDTH        = 0.15 * 0.1
WHEEL_LENGTH       = 0.4  * 0.1


class CarDynamics(Dynamics):
    """
    Car movement dynamics.
    """
    def __init__(self, Q=None):
        """
          Q:
             System noise covariance (numpy nd-array)
        """
        super(CarDynamics, self).__init__()

        self.Q = Q
        
        # Jacobian calculation with automatic differentiation
        self.x = jacobian(self.value, 0)
        self.u = jacobian(self.value, 1)

    def value(self, x, u):
        # Controls
        w  = u[0] # Front wheel angle
        a  = u[1] # Front wheel acceleration
    
        o  = x[2] # Car angle
        v  = x[3] # Velocity
    
        # Change in car angle
        d_angle = v / CAR_AXLES_DISTANCE * np.tan(w)

        dx = np.array([v * np.cos(o),
                       v * np.sin(o),
                       d_angle,
                       a],
                      dtype=np.float32)

        if self.Q is not None:
            # Add system noise
            dx += np.random.multivariate_normal(np.zeros_like(x), self.Q)
        
        return dx

    @property
    def x_dim(self):
        return 4

    @property
    def u_dim(self):
        return 2


class CarRenderer(Renderer):
    """
    Car renderer.
    Used for MPCEnv when image output option is used.
    """
    def __init__(self, image_width=256):
        super(CarRenderer, self).__init__()
        self.image_width = image_width

    def draw_rotated_rect(self, image, x, y, w, h, angle, color):
        rect = ((x,y), (h, w), angle / np.pi * 180.0)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        image = cv2.drawContours(image, [box], 0, color, 1)
        return image

    def render(self, x, u):
        image = np.ones((self.image_width, self.image_width, 3), dtype=np.float32)
        
        render_scale = self.image_width / 2.0
        render_offset = self.image_width / 2.0

        agent_angle = x[2]
        
        # Center pos of the car
        agent_cener_x = x[0] + np.cos(agent_angle) * CAR_LENGTH/2
        agent_cener_y = x[1] + np.sin(agent_angle) * CAR_LENGTH/2

        # Render car body
        image = self.draw_rotated_rect(image,
                                       agent_cener_x * render_scale + render_offset,
                                       agent_cener_y * render_scale + render_offset,
                                       CAR_WIDTH * render_scale,
                                       CAR_LENGTH * render_scale,
                                       agent_angle,
                                       (0,0,1))

        # Render wheels
        wheel_angle = agent_angle + u[0]
        wheel_pos0_x = x[0] + np.cos(agent_angle) * CAR_AXLES_DISTANCE + \
                       -np.sin(agent_angle) * CAR_WIDTH/2
        wheel_pos0_y = x[1] + np.sin(agent_angle) * CAR_AXLES_DISTANCE + \
                       +np.cos(agent_angle) * CAR_WIDTH/2
        wheel_pos1_x = x[0] + np.cos(agent_angle) * CAR_AXLES_DISTANCE + \
                       +np.sin(agent_angle) * CAR_WIDTH/2
        wheel_pos1_y = x[1] + np.sin(agent_angle) * CAR_AXLES_DISTANCE + \
                       -np.cos(agent_angle) * CAR_WIDTH/2

        wheel_pos2_x = x[0] + \
                       -np.sin(agent_angle) * CAR_WIDTH/2
        wheel_pos2_y = x[1] + \
                       +np.cos(agent_angle) * CAR_WIDTH/2
        wheel_pos3_x = x[0] + \
                       +np.sin(agent_angle) * CAR_WIDTH/2
        wheel_pos3_y = x[1] + \
                       -np.cos(agent_angle) * CAR_WIDTH/2
        
        image = self.draw_rotated_rect(image,
                                       wheel_pos0_x * render_scale + render_offset,
                                       wheel_pos0_y * render_scale + render_offset,
                                       WHEEL_WIDTH * render_scale,
                                       WHEEL_LENGTH * render_scale,
                                       wheel_angle,
                                       (0,0,1))
        image = self.draw_rotated_rect(image,
                                       wheel_pos1_x * render_scale + render_offset,
                                       wheel_pos1_y * render_scale + render_offset,
                                       WHEEL_WIDTH * render_scale,
                                       WHEEL_LENGTH * render_scale,
                                       wheel_angle,
                                       (0,0,1))
        image = self.draw_rotated_rect(image,
                                       wheel_pos2_x * render_scale + render_offset,
                                       wheel_pos2_y * render_scale + render_offset,
                                       WHEEL_WIDTH * render_scale,
                                       WHEEL_LENGTH * render_scale,
                                       agent_angle,
                                       (0,0,1))
        image = self.draw_rotated_rect(image,
                                       wheel_pos3_x * render_scale + render_offset,
                                       wheel_pos3_y * render_scale + render_offset,
                                       WHEEL_WIDTH * render_scale,
                                       WHEEL_LENGTH * render_scale,
                                       agent_angle,
                                       (0,0,1))
        return image


class CarObstacle(object):
    def __init__(self, pos, is_good):
        self.pos = pos
        self.is_good = is_good
        self.radius = 0.1
        self.hit_time = -1.0

    def apply_state(self, x, t):
        if self.hit_time >= 0.0:
            # Already hit
            return
        elif self.contains(x):
            # If agent pos was in this region
            self.hit_time = t

    def contains(self, x):
        agent_pos = x[:2]
        d = agent_pos - self.pos
        return np.sum(d**2) < self.radius**2

    def calc_rate(self, t):
        if self.hit_time >= 0.0:
            # TODO: calculate rate
            return 0.0
        else:
            return 1.0


class CarCost(Cost):
    """
    Car cost function for obstacle avoidance.
    """
    def __init__(self, obstacles):
        super(CarCost, self).__init__()
        self.obstacles = obstacles
        
        self.x  = grad(self.value, 0)
        self.u  = grad(self.value, 1)
        self.xx = jacobian(self.x, 0)
        self.uu = jacobian(self.u, 1)
        self.ux = jacobian(self.u, 0)

    def clone(self):
        cost = CarCost(copy.deepcopy(self.obstacles))
        return cost

    def apply_state(self, x, t):
        for obstacle in self.obstacles:
            obstacle.apply_state(x, t)

    def value(self, x, u, t):
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        def sharp_edge(x):
            #return 1.0 - np.exp(- np.power(10.0 * x, 2))
            return 1.0 - np.exp(- np.power(1.0 * x, 2))

        x_cost = 0.0
        for obstacle in self.obstacles:
            dx = x[0] - obstacle.pos[0]
            dy = x[1] - obstacle.pos[1]
            distance = np.sqrt(dx * dx + dy * dy)
            if obstacle.is_good:
                # More far = Larger cost (Nearer is better)
                #x_cost += 1.0 * sharp_edge(1.0 * distance) * obstacle.calc_rate(x)
                x_cost += 1.0 * sigmoid(1.0 * distance) * obstacle.calc_rate(x)
            else:
                # More far = Smaller cost (Mor far is better
                x_cost -= 0.5 * sigmoid(1.0 * distance) * obstacle.calc_rate(x)
                #x_cost -= 0.5 * sharp_edge(1.0 * distance) * obstacle.calc_rate(x)

        if u is not None:
            # Running cost
            u_cost = 0.01 * (u[0]**2) + 0.5 * (u[1]**2)
            return x_cost + u_cost
        else:
            # Terminal cost
            return x_cost
