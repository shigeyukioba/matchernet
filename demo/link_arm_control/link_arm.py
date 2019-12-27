# -*- coding: utf-8 -*-
import autograd.numpy as np
from autograd import jacobian
import math
import pyglet
from pyglet.gl import *
from ctypes import POINTER

from matchernet import Dynamics, Renderer

from graphics import FrameBuffer, MultiSampleFrameBuffer
from objmesh import ObjMesh, load_texture_in_data_dir
from camera import Camera

BG_COLOR = np.array([0.45, 0.82, 1.0, 1.0])
WHITE_COLOR = np.array([1.0, 1.0, 1.0])

# Camera vertical field of view angle (degree)
CAMERA_FOV_Y = 50


def rad2deg(radian):
    degree = radian * -180.0 / np.pi
    return degree


class LinkArmDynamics(Dynamics):
    """ Two link arm dynanamics.
    
    Adopted from https://courses.ideate.cmu.edu/16-375/f2018/text/lib/double-pendulum.html
    """
    def __init__(self):
        super(LinkArmDynamics, self).__init__()

        self.l1   = 1.0    # proximal link length, link1
        self.l2   = 1.0    # distal link length, link2
        self.lc1  = 0.5    # distance from proximal joint to link1 COM
        self.lc2  = 0.5    # distance from distal joint to link2 COM
        self.m1   = 1.0    # link1 mass
        self.m2   = 1.0    # link2 mass
        self.I1   = (self.m1 * self.l1**2) / 12  # link1 moment of inertia
        self.I2   = (self.m2 * self.l2**2) / 12  # link2 moment of inertia
        self.gravity  = -9.81
        
        # Jacobian calculation with automatic differentiation
        self.x = jacobian(self.value, 0)
        self.u = jacobian(self.value, 1)

    def value(self, x, u):
        q1  = x[0]
        q2  = x[1]
        qd1 = x[2]
        qd2 = x[3]
        
        LC1 = self.lc1
        LC2 = self.lc2
        L1  = self.l1
        M1  = self.m1
        M2  = self.m2

        d11 = M1 * LC1 * LC1  + M2 * (L1 * L1 + LC2 * LC2 + 2 * L1 * LC2 * np.cos(q2)) + self.I1 + self.I2
        d12 = M2 * (LC2 * LC2 + L1 * LC2 * np.cos(q2)) + self.I2
        d21 = d12
        d22 = M2 * LC2 * LC2  + self.I2
        
        h1 = -M2 * L1 * LC2 * np.sin(q2) * qd2 * qd2 - 2 * M2 * L1 * LC2 * np.sin(q2) * qd2 * qd1
        h2 =  M2 * L1 * LC2 * np.sin(q2) * qd1 * qd1

        phi1 = -M2 * LC2 * self.gravity * np.sin(q1+q2) - (M1 * LC1 + M2 * L1) * \
               self.gravity * np.sin(q1)
        phi2 = -M2 * LC2 * self.gravity * np.sin(q1 + q2)

        # now solve the equations for qdd:
        #  d11 qdd1 + d12 qdd2 + h1 + phi1 = u1
        #  d21 qdd1 + d22 qdd2 + h2 + phi2 = u2
        
        rhs1 = u[0] - h1 - phi1
        rhs2 = u[1] - h2 - phi2
        
        # Apply Cramer's Rule to compute the accelerations using
        # determinants by solving D qdd = rhs.  
        # First compute the denominator as the determinant of D:
        denom = (d11 * d22) - (d21 * d12)

        dx = np.array([qd1,
                       qd2,
                       ((rhs1 * d22 ) - (rhs2 * d12)) / denom,
                       (( d11 * rhs2) - (d21  * rhs1)) / denom],
                      dtype=np.float32)

        return dx

    @property
    def x_dim(self):
        return 4

    @property
    def u_dim(self):
        return 2

# Whether to use multi sample frame buffer    
USE_MULTI_SAMPLE_FRAME_BUFFER = False


class LinkArmRenderer(Renderer):
    def __init__(self, eye_from, eye_to, image_width=128):
        super(LinkArmRenderer, self).__init__()
        
        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(width=1, height=1, visible=False)

        if USE_MULTI_SAMPLE_FRAME_BUFFER:
            self.frame_buffer = MultiSampleFrameBuffer(image_width, image_width, 2)
        else:
            self.frame_buffer = FrameBuffer(image_width, image_width)
        
        self.camera = Camera(eye_from=eye_from, eye_to=eye_to)

        self.link_mesh = ObjMesh.get("link0")
        self.joint_mesh = ObjMesh.get("joint0")

        self.link_textures  = [load_texture_in_data_dir("link{}".format(i)) for i in range(2)]
        self.joint_textures = [load_texture_in_data_dir("joint{}".format(i)) for i in range(3)]

    def render(self, x, u=None):
        image = self.render_sub(self.frame_buffer, x)

        # Change upside-down
        image = np.flip(image, 0)
        
        return image

    def render_link(self, pos, angle, scale, index):
        angle_degree = rad2deg(angle)
        
        glPushMatrix()
        glTranslatef(*pos)
        glScalef(scale, scale, scale)
        glRotatef(angle_degree, 0, 0, 1)
        texture = self.link_textures[index]
        self.link_mesh.render(texture)
        glPopMatrix()

    def render_joint(self, pos, scale, index):
        glPushMatrix()
        glTranslatef(*pos)
        glScalef(scale, scale, scale)
        texture = self.joint_textures[index]
        self.joint_mesh.render(texture)
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
        glLightfv(GL_LIGHT0, GL_AMBIENT, vec(0.9, 0.9, 0.9, 1.0))
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

        # Render links
        link_length = 0.85
        link_scale = 0.5
        joint_scale = 0.15
        
        joint_pos = np.array([0,0,0], dtype=np.float32)
        self.render_joint(joint_pos, joint_scale, 0)
        accum_angle = 0.0

        link_size = len(x) // 2
        for i in range(link_size):
            angle = x[i]
            
            accum_angle += angle
            
            link_vec = np.array([np.sin(accum_angle), -np.cos(accum_angle), 0.0],
                                dtype=np.float32)
            center_pos = joint_pos + link_vec * link_length * 0.5
            joint_pos = joint_pos + link_vec * link_length
            
            self.render_link(center_pos, -accum_angle, link_scale, i)
            self.render_joint(joint_pos, joint_scale, i+1)            
        
        return frame_buffer.read()
