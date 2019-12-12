import logging
import numpy as np
import matplotlib.pyplot as plt
from brica import Component, VirtualTimeScheduler, Timing

from matchernet_py_001.ekf import BundleEKFContinuousTime, MatcherEKF
from matchernet_py_001 import fn
from matchernet_py_001 import observer
from matchernet_py_001.state_space_model_2d import StateSpaceModel2Dim
from matchernet_py_001 import utils
from matchernet_py_001.utils import print_flush

logger = logging.getLogger(__name__)
formatter = '[%(asctime)s] %(module)s.%(funcName)s %(levelname)s -> %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)


def visualize_bundle_rec(b, y_rec=None):

    mu_rec = b.record["mu"]
    sigma_rec = b.record["diagSigma"]
    time_stamp = b.record["time_stamp"]

    logger.debug("mu_rec={}".format(mu_rec))
    logger.debug("sigma_rec={}".format(sigma_rec))

    plt.subplot(221)
    plt.plot(mu_rec[:, 0], mu_rec[:, 1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory")

    plt.subplot(222)
    yd = mu_rec[:, 0] - sigma_rec[:, 0]
    yu = mu_rec[:, 0] + sigma_rec[:, 0]
    plt.fill_between(time_stamp, yd, yu, facecolor='y', alpha=0.5)
    plt.plot(time_stamp, mu_rec[:, 0])
    plt.ylabel("X")
    plt.xlabel("time")

    plt.subplot(224)
    yd = mu_rec[:, 1] - sigma_rec[:, 1]
    yu = mu_rec[:, 1] + sigma_rec[:, 1]
    plt.fill_between(time_stamp, yd, yu, facecolor='y', alpha=0.5)
    plt.plot(time_stamp, mu_rec[:, 1])
    plt.ylabel("Y")
    plt.xlabel("time")

    if y_rec is None:
        return
    else:
        plt.subplot(221)
        plt.scatter(y_rec[:, 0], y_rec[:, 1], s=2)
        plt.subplot(222)
        plt.scatter(time_stamp, y_rec[:, 0], s=2)
        plt.subplot(224)
        plt.scatter(time_stamp, y_rec[:, 1], s=2)


mu0 = np.array([0, 1.0], dtype=np.float32)
A0 = np.array([[-0.1, 2], [-2, -0.1]], dtype=np.float32)
ey2 = np.eye(2, dtype=np.float32)

def test_BundleEKFContinuousTime01(dt, n_steps):
    b = BundleEKFContinuousTime("B0", 2, fn.LinearFn(A0))
    b.dt = dt
    b.logger_state()
    b.state.data["mu"] = mu0

    dummy_input = {} # list of matchers (#matcher=0)
    for i in range(n_steps):
        b(dummy_input)

    visualize_bundle_rec(b)


def test_bundle_and_observer(dt, n_steps, y_rec):
    b0 = observer.Observer("b0", y_rec)
    b1 = BundleEKFContinuousTime("b1", 2, fn.LinearFn(A0))
    b1.state.data["mu"] = mu0
    b1.dt = dt
    dummy_input = {}
    for i in range(n_steps):
        b0(dummy_input)
        b1(dummy_input)
        y = b0.get_state()
        if i == 0:
            y_rec2 = y
        else:
            y_rec2 = np.vstack((y_rec2, y))

    visualize_bundle_rec(b1, y_rec2)


def test_MatcherEKF01(dt, n_steps, y_rec):
    """
    Run a matchernet of the following structure
    b0 --- m01 --- b1
    """

    b1 = observer.Observer("b1", y_rec)
    b1.obs_noise_covariance = 2 * ey2

    b0 = BundleEKFContinuousTime("b0", 2, fn.LinearFn(A0))
    b0.state.data["mu"] = mu0
    b0.dt = dt
    b0.state.data["mu"][1] = 2
    b0.state.data["Sigma"] = 2 * ey2

    m01 = MatcherEKF("m01", b0, b1)


    if _with_brica is False:
        for i in range(n_steps):
            print_flush("Step {}/{} with brica".format(i, n_steps))
            inputs_to_m01 = {"b0": b0.state, "b1": b1.state}
            results = m01(inputs_to_m01)
            inputs_to_b0 = {"m01": results["b0"]}
            s0 = b0(inputs_to_b0)
            inputs_to_b1 = {"m01": results["b1"]}
            s1 = b1(inputs_to_b1)
    else:
        s = VirtualTimeScheduler()

        bt = Timing(0, 1, 1)
        bm = Timing(1, 1, 1)

        s.add_component(b0.component, bt)
        s.add_component(b1.component, bt)
        s.add_component(m01.component, bm)

        for i in range(n_steps):
            print_flush("Step {}/{} with brica".format(i, n_steps+1))
            s.step()

    # visualize_bundle_rec(b0, y_rec)


if __name__ == '__main__':
    # if False:
    #     print("===Starting UnitTest01===")
    #     print("-- A simple test of Bundle with no Matcher")
    #     plt.figure(1)
    #     test_BundleEKFContinuousTime01(dt, n_steps)
    #     plt.pause(0.2)

    dt = 0.02
    n_steps = 500

    # preparing a list of three simulation sequences
    yrecs = []
    for i in range(1):
        sm=StateSpaceModel2Dim(
            n_dim=2,
            A=np.array([[-0.1, 2], [-2, -0.1]], dtype=np.float32),
            g=fn.LinearFn(utils.zeros(2)),
            sigma_w=0.1,
            sigma_z=0.1,
            x=np.array([0, 0], dtype=np.float32),
            y=utils.zeros((1, 2))
        )
        sm.A = A0
        sm.x = mu0
        sm.dt = dt
        (x_rec, y_rec) = sm.simulation(n_steps, dt)
        yrecs.append(y_rec)

    plt.figure(0)
    for i in range(1):
        plt.subplot(2, 2, i+1)
        plt.scatter(yrecs[i][:, 0], yrecs[i][:, 1], s=2)
        plt.title("sequence {}".format(i))
    plt.pause(1)

    if False:
        logger.info("-- A simple test of Bundle with Observer with no Matcher")
        plt.figure(2)
        test_bundle_and_observer(dt, n_steps, y_rec)

    if True:
        _with_brica = True
        logger.info("-- A simple test to link a Bundle, a Observer, and a Matcher")
        logger.info("-- with brica: {}".format(_with_brica))
        plt.figure(4)
        test_MatcherEKF01(dt, n_steps, y_rec)

    if False:
        _with_brica = True
        logger.info("-- A simple test to link a Bundle, a Observer, and a Matcher")
        logger.info("-- with brica: {}".format(_with_brica))
        plt.figure(3)
        test_MatcherEKF01(dt, n_steps, y_rec)

    plt.pause(5)
