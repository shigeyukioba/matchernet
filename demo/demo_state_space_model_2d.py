import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt

from matchernet_py_001.state_space_model_2d import StateSpaceModel2Dim

printDebug = True
def demo_state_space_model_2dim(numSteps):
    dt = 0.1
    ssm = StateSpaceModel2Dim(dt)
    xrec, yrec = ssm.simulation(dt, numSteps)
    if printDebug is True:
        print('x: ',xrec)
        print('y: ',yrec)

    plt.subplot(211)
    timestamp=np.array(range(0,numSteps))
    plt.plot(timestamp,xrec[:,0])
    plt.scatter(timestamp,yrec[:,0],s=1)
    plt.title("test_state_space_model_2Dim")
    plt.ylabel("X")
    plt.subplot(212)
    plt.plot(timestamp,xrec[:,1])
    plt.scatter(timestamp,yrec[:,1],s=1)
    plt.ylabel("Y")
    plt.show()

if __name__ == '__main__':
    demo_state_space_model_2dim(500)
