# -*- coding: utf-8 -*-
import numpy as np

from vrep_env import VREPEnv

def main():
    env = VREPEnv(direct=True)
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
