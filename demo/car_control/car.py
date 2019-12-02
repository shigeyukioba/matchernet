# -*- coding: utf-8 -*-
import autograd.numpy as np
from autograd import jacobian
import cv2

from matchernet import Dynamics, Renderer


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
    def __init__(self, dt, Q=None):
        """
          dt
            timestap (second)
          Q:
             System noise covariance (numpy nd-array)
        """
        super(CarDynamics, self).__init__(dt)

        self.Q = Q
        
        # Jacobian calculation with automatic differentiation
        self.x = jacobian(self.value, 0)
        self.u = jacobian(self.value, 1)

    def value(self, x, u):
        # Controls
        w  = u[0] # Front wheel angle
        a  = u[1] # Front wheel acceleration
    
        o  = x[2] # Car angle
        v  = x[3] # Front wheel velocity
    
        # Front wheel rolling distance
        f  = v * self.dt
    
        # Back wheel rolling distance
        b = CAR_AXLES_DISTANCE + f * np.cos(w) - \
            np.sqrt(CAR_AXLES_DISTANCE ** 2 - (f * np.sin(w)) ** 2)
    
        # Change in car angle
        d_angle = np.arcsin(np.sin(w) * f/CAR_AXLES_DISTANCE)
    
        # Change in state
        dx = np.array([b*np.cos(o), b*np.sin(o), d_angle, a*self.dt],
                      dtype=np.float32)
    
        # New state
        x_new = x + dx

        if self.Q is not None:
            # Add system noise
            x_new += np.random.multivariate_normal(np.zeros_like(x_new),
                                                    self.Q * self.dt)
        
        # Limit x,y pos range For debugging)
        x_min = np.array([-1, -1, -np.inf, -np.inf], dtype=np.float32)
        x_max = np.array([ 1,  1,  np.inf,  np.inf], dtype=np.float32)
        x_new = np.clip(x_new, x_min, x_max)
        return x_new

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
    def __init__(self):
        super(CarRenderer, self).__init__()

    def draw_rotated_rect(self, image, x, y, w, h, angle, color):
        rect = ((x,y), (h, w), angle / np.pi * 180.0)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        image = cv2.drawContours(image, [box], 0, color, 1)
        return image

    def render(self, image, x, u):
        image_width = image.shape[1]
        
        render_scale = image_width / 2.0
        render_offset = image_width / 2.0

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
