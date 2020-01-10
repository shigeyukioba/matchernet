# -*- coding: utf-8 -*-
import numpy as np
from abr_control.arms import jaco2 as arm
from abr_control.interfaces import VREP


class VREPEnv(object):
    def __init__(self, direct=False):
        self.robot_config = arm.Config(use_cython=False, hand_attached=True)
        
        # Timestap is fixed and specified in .ttt file
        self.dt = 0.005
        self.direct = direct

        if direct:
            self.interface = VREP(robot_config=self.robot_config, dt=self.dt)
            self.interface.connect()
        else:
            self.interface = None

        self.requested_reset_q = None
        self.requested_target_shadow_q = None
        self.close_requested = False

    @property
    def u_dim(self):
        return self.robot_config.N_JOINTS

    @property
    def x_dim(self):
        return self.robot_config.N_JOINTS * 2

    def reset_angles(self, q):
        if self.direct:
            self._reset_angles(q)
        else:
            self.requested_reset_q = q

    def _reset_angles(self, q):
        self.interface.send_target_angles(q)            

    def set_target_shadow(self, q, direct=False):
        if self.direct:
            self._set_target_shadow(q)
        else:
            self.requested_target_shadow_q = q

    def _set_target_shadow(self, q):
        names = ['joint%i_shadow' % i for i in range(self.robot_config.N_JOINTS)]
        joint_handles = []
        for name in names:
            interface.get_xyz(name)  # this loads in the joint handle
            joint_handles.append(self.interface.misc_handles[name])
        self.interface.send_target_angles(q, joint_handles)            

    def close(self, direct=False):
        if self.direct:
            self._close()
        else:
            self.close_requested = True

    def _close(self):
        self.interface.disconnect()
        
    def step(self, action):
        if self.interface is None:
            self.interface = VREP(robot_config=self.robot_config, dt=self.dt)
            self.interface.connect()

        if self.requested_target_shadow_q is not None:
            self._set_target_shadow(self.requested_target_shadow_q)
            self.requested_target_shadow_q = None

        if self.requested_reset_q is not None:
            self._reset_angles(self.requested_reset_q)
            self.requested_reset_q = None
        
        # Send torq to VREP
        self.interface.send_forces(action)
        
        feedback = self.interface.get_feedback()

        # Get joint angle and velocity feedback
        q = feedback['q']
        dq = feedback['dq']

        # Concatenate q and dq
        state = np.hstack((q, dq)) # (12,)
        reward = 0.0

        if self.close_requested:
            self._close()
            self.close_requested = False
        
        return state, reward
