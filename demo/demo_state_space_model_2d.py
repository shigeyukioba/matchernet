import logging
import numpy as np
import matplotlib.pyplot as plt

from matchernet_py_001 import fn
from matchernet_py_001.state_space_model_2d import StateSpaceModel2Dim
from matchernet_py_001 import utils

logger = logging.getLogger(__name__)
def demo_state_space_model_2dim(n_dim, A, g, sigma_w, sigma_z, x, y, n_steps, dt):
    ssm = StateSpaceModel2Dim(n_dim, A, g, sigma_w, sigma_z, x, y)
    x_series, y_series = ssm.simulation(n_steps, dt)
    logger.info('x: {}'.format(x_series))
    logger.info('y: {}'.format(y_series))

    plt.subplot(211)
    timestamp=np.array(range(0, n_steps))
    plt.plot(timestamp, x_series[:, 0])
    plt.scatter(timestamp, y_series[:, 0], s=1)
    plt.title("test_state_space_model_2Dim")
    plt.ylabel("X")
    plt.subplot(212)
    plt.plot(timestamp, x_series[:, 1])
    plt.scatter(timestamp, y_series[:, 1], s=1)
    plt.ylabel("Y")
    plt.show()

if __name__ == '__main__':
    formatter = '[%(asctime)s] %(module)s.%(funcName)s %(levelname)s -> %(message)s'
    logging.basicConfig(level=logging.INFO, format=formatter)

    demo_state_space_model_2dim(
        n_dim=2,
        A=np.array([[-0.1, 2], [-2, -0.1]], dtype=np.float32),
        g=fn.LinearFn(utils.zeros(2)),
        sigma_w=0.1,
        sigma_z=0.1,
        x=np.array([0, 0], dtype=np.float32),
        y=utils.zeros((1, 2)),
        n_steps=500,
        dt=0.1
    )
