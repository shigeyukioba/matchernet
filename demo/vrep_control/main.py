import numpy as np
import traceback

from abr_control.arms import jaco2 as arm
from abr_control.interfaces import VREP


class VREPEnv(object):
    def __init__(self):
        self.robot_config = arm.Config(use_cython=False,
                                       hand_attached=True)

        # Timestap is fixed and specified in .ttt file
        self.dt = 0.005
        
        # create interface and connect
        self.interface = VREP(robot_config=self.robot_config, dt=self.dt)
        self.interface.connect()

    @property
    def u_dim(self):
        return self.robot_config.N_JOINTS

    @property
    def x_dim(self):
        return self.robot_config.N_JOINTS * 2

    def reset_angles(self, q):
        self.interface.send_target_angles(q)

    def set_target_shadow(self, q):
        names = ['joint%i_shadow' % i for i in range(self.robot_config.N_JOINTS)]
        joint_handles = []
        for name in names:
            interface.get_xyz(name)  # this loads in the joint handle
            joint_handles.append(self.interface.misc_handles[name])
        self.interface.send_target_angles(q, joint_handles)
        
    def step(self, action):
        # Send torq to VREP
        self.interface.send_forces(action)
        
        feedback = self.interface.get_feedback()

        # Get joint angle and velocity feedback
        q = feedback['q']
        dq = feedback['dq']

        # Concatenate q and dq
        state = np.hstack((q, dq)) # (12,)
        reward = 0.0
        return state, reward

    def close(self):
        self.interface.disconnect()


def main():
    env = VREPEnv()
    u_dim = env.u_dim # 6

    default_angles = np.array([-3.48837466e-06,
                               2.75779843e+00,
                               2.70459056e+00,
                               -1.57118285e+00,
                               1.70024168e-02,
                               3.07200003e+00])
    q0 = default_angles
    env.reset_angles(q0)
    
    for i in range(10):
        u = np.zeros((u_dim,), dtype=np.float32)
        state, _ = env.step(u)
        print(state)

    env.close()

if __name__ == '__main__':
    main()
