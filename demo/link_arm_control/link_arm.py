# -*- coding: utf-8 -*-
import sys
import numpy as np
import math

import pyglet
from pyglet.gl import *
from ctypes import POINTER

from graphics import FrameBuffer, MultiSampleFrameBuffer
from objmesh import ObjMesh
from camera import Camera

BG_COLOR = np.array([0.45, 0.82, 1.0, 1.0])
WHITE_COLOR = np.array([1.0, 1.0, 1.0])

# Camera vertical field of view angle (degree)
CAMERA_FOV_Y = 50


def rad2deg(radian):
    degree = radian * -180.0 / math.pi
    return degree


class LinkArmRenderer(object):
    def __init__(self, eye_from, eye_to, buffer_width=128):
        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(width=1, height=1, visible=False)
        #self.frame_buffer = FrameBuffer(buffer_width, buffer_width)
        self.frame_buffer = MultiSampleFrameBuffer(buffer_width, buffer_width, 4)
        self.camera = Camera(eye_from=eye_from, eye_to=eye_to)
        
        self.arm_mesh = ObjMesh.get("arm0")
        self.link_mesh = ObjMesh.get("link0")

    def render(self, x):
        image = self.render_sub(self.frame_buffer, x)

        # Change upside-down
        image = np.flip(image, 0)
        
        return image

    def render_arm(self, pos, angle, scale):
        angle_degree = rad2deg(angle)
        
        glPushMatrix()
        glTranslatef(*pos)
        glScalef(scale, scale, scale)
        glRotatef(angle_degree, 0, 0, 1)
        self.arm_mesh.render()
        glPopMatrix()

    def render_link(self, pos, scale):
        glPushMatrix()
        glTranslatef(*pos)
        glScalef(scale, scale, scale)
        self.link_mesh.render()
        glPopMatrix()        

    def render_sub(self, frame_buffer, x):
        self.shadow_window.switch_to()

        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*BG_COLOR)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        def vec(*args):
            return (GLfloat * len(args))(*args)
        
        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            CAMERA_FOV_Y,
            frame_buffer.width / float(frame_buffer.height),
            0.04,  # near plane
            100.0  # far plane
        )

        # Apply camera angle
        glMatrixMode(GL_MODELVIEW)
        m = self.camera.get_inv_mat()
        glLoadMatrixf(m.get_raw_gl().ctypes.data_as(POINTER(GLfloat)))

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, vec(1.0, -1.0, -1.0, 0.0));
        glLightfv(GL_LIGHT0, GL_AMBIENT, vec(0.8, 0.8, 0.8, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, vec(0, 0, 0, 0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(1.0, 1.0, 1.0, 1.0))
        glEnable(GL_NORMALIZE)
        #glShadeModel(GL_SMOOTH)
        
        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # For each object
        glColor3f(*WHITE_COLOR)

        # Render arms
        #..
        arm_length = 0.85
        arm_scale = 0.5
        link_scale = 0.15
        
        link_pos = np.array([0,0,0], dtype=np.float32)
        accum_angle = 0.0
        for i, angle in enumerate(x):
            self.render_link(link_pos, link_scale)
            
            accum_angle += angle
            
            link_vec = np.array([np.sin(accum_angle), -np.cos(accum_angle), 0.0],
                                dtype=np.float32)
            center_pos = link_pos + link_vec * arm_length * 0.5
            link_pos = link_pos + link_vec * arm_length
            self.render_arm(center_pos, -accum_angle, arm_scale)
        #..
        
        return frame_buffer.read()
