import logging
import numpy as np
import matplotlib.pyplot as plt

from matchernet_py_001.state_space_model_2d import StateSpaceModel2Dim

logger = logging.getLogger(__name__)
def demo_state_space_model_2dim(numSteps):
    dt = 0.1
    ssm = StateSpaceModel2Dim(dt)
    x_series, y_series = ssm.simulation(dt, numSteps)
    logger.info('x: {}'.format(x_series))
    logger.info('y: {}'.format(y_series))

    plt.subplot(211)
    timestamp=np.array(range(0,numSteps))
    plt.plot(timestamp,x_series[:,0])
    plt.scatter(timestamp,y_series[:,0],s=1)
    plt.title("test_state_space_model_2Dim")
    plt.ylabel("X")
    plt.subplot(212)
    plt.plot(timestamp,x_series[:,1])
    plt.scatter(timestamp,y_series[:,1],s=1)
    plt.ylabel("Y")
    plt.show()

if __name__ == '__main__':
    formatter = '[%(asctime)s] %(module)s.%(funcName)s %(levelname)s -> %(message)s'
    logging.basicConfig(level=logging.INFO, format=formatter)
    demo_state_space_model_2dim(500)
