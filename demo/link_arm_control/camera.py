# -*- coding: utf-8 -*-
import sys
import numpy as np

from geom import Matrix4


class Camera(object):
    """ 3D camera class. """
    def __init__(self, eye_from, eye_to):
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.m = self.get_lookat_mat(eye_from, eye_to, up)
        self.m_inv = self.m.invert()

    def get_lookat_mat(self, eye_from, eye_to, up):
        def normalize_vec(v):
            v /= np.linalg.norm(v)
            return v
        
        forward = eye_to - eye_from;
        forward = normalize_vec(forward)
    
        side = np.cross(forward, up)
        side = normalize_vec(side)
        
        new_up = np.cross(side, forward)

        m = Matrix4()
        m.m[0,0] = side[0];
        m.m[1,0] = side[1];
        m.m[2,0] = side[2];
    
        m.m[0,1] = new_up[0];
        m.m[1,1] = new_up[1];
        m.m[2,1] = new_up[2];

        m.m[0,2] = -forward[0];
        m.m[1,2] = -forward[1];
        m.m[2,2] = -forward[2];

        m.m[0,3] = eye_from[0];
        m.m[1,3] = eye_from[1];
        m.m[2,3] = eye_from[2];
        return m

    def get_inv_mat(self):
        """ Get invererted camera matrix

        Returns:
          numpy ndarray: inverted camera matrix
        """
        return self.m_inv
