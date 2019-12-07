import numpy as np
import matplotlib.pyplot as plt
from brica import Component, VirtualTimeScheduler, Timing

from matchernet_py_001.ekf import BundleEKFContinuousTime, MatcherEKF
from matchernet_py_001 import fn
from matchernet_py_001 import observer
from matchernet_py_001.state_space_model_2d import StateSpaceModel2Dim
from matchernet_py_001 import utils
from matchernet_py_001.utils import print1, print2, print3, print4, print_flush

_with_brica = True

#=======================================================================
#  Visualization functions for the matchernet_ekf.py
#=======================================================================
def visualize_bundle_rec(b, y_rec=None):

    mu_rec = b.record["mu"]
    sigma_rec = b.record["diagSigma"]
    time_stamp = b.record["time_stamp"]
    n_steps = mu_rec.shape[0]
    timestamp = np.array(range(0, n_steps))

    print3("mu_rec.shape={}".format(mu_rec.shape))
    print3("mu_rec.dtype={}".format(mu_rec.dtype))
    print3("sigma_rec.shape={}".format(sigma_rec.shape))
    print3("sigma_rec.dtype={}".format(sigma_rec.dtype))
    print3("timestamp.shape={}".format(timestamp.shape))
    print4("mu_rec={}".format(mu_rec))
    print4("sigma_rec={}".format(sigma_rec))
    print4("timestamp={}".format(timestamp))

    plt.subplot(221)
    plt.plot(mu_rec[:, 0], mu_rec[:, 1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory")


    plt.subplot(222)
    yd = mu_rec[:, 0] - sigma_rec[:, 0]
    yu = mu_rec[:, 0] + sigma_rec[:, 0]
    print4("sigma_rec[:,0]={}".format(sigma_rec[:, 0]))
    print4("mu_rec[:,0]={}".format(mu_rec[:, 0]))
    print3("mu_rec[:,0].shape={}".format(mu_rec[:, 0].shape))
    print3("sigma_rec[:,0].shape={}".format(sigma_rec[:, 0].shape))
    print3("mu_rec[:,1].shape={}".format(mu_rec[:, 1].shape))
    print3("sigma_rec[:,1].shape={}".format(sigma_rec[:, 1].shape))
    print3("yu.shape={}".format(yu.shape))
    plt.fill_between(time_stamp, yd, yu, facecolor='y', alpha=0.5)
    plt.plot(time_stamp, mu_rec[:, 0])
    plt.ylabel("X")
    plt.xlabel("time")

    plt.subplot(224)
    yd = mu_rec[:, 1] - sigma_rec[:, 1]
    yu = mu_rec[:, 1] + sigma_rec[:, 1]
    plt.fill_between(time_stamp, yd, yu, facecolor='y', alpha=0.5)
    plt.plot(time_stamp, mu_rec[:, 1])
    #plt.scatter(range(0, 5000-1), mu_rec[:, 0])
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


#=======================================================================
#  Test functions for the matchernet_ekf.py
#=======================================================================

mu0 = np.array([0, 1.0], dtype=np.float32)
A0 = np.array([[-0.1, 2], [-2, -0.1]], dtype=np.float32)
ey2 = np.eye(2, dtype=np.float32)

def test_BundleEKFContinuousTime01(dt, n_steps):
    # test of BundleEKFContinuousTime with a two dimensional linear dynamics
    # dx = dt * F * x + Q * dw
    b = BundleEKFContinuousTime("B0", 2)
    b.dt = dt
    b.print_state()
    b.f = fn.LinearFn(A0)
    b.state.data["mu"] = mu0

    dummy_input = {} # list of matchers (#matcher=0)
    for i in range(n_steps):
        b(dummy_input)

    visualize_bundle_rec(b)


def test_bundle_and_observer(dt, n_steps, y_rec):
    b0 = observer.Observer("b0", y_rec)
    b1 = BundleEKFContinuousTime("b1", 2)
    b1.f = fn.LinearFn(A0)
    b1.state.data["mu"] = mu0
    b1.dt = dt
    dummy_input = {} # list of matchers (#matcher=0)
    for i in range(n_steps):
        b0(dummy_input)
        b1(dummy_input)
        y = b0.get_state()
        if i == 0:
            y_rec2 = y
        else:
            y_rec2 = np.concatenate((y_rec2, y), axis=0)

    timestamp = np.array(range(0, n_steps))
    visualize_bundle_rec(b1, y_rec2)


def test_MatcherEKF01(dt, n_steps, y_rec):
    '''
    Run a matchernet of the following structure
    b0 --- m01 --- b1
    '''

    b1 = observer.Observer("b1", y_rec)
    b1.obs_noise_covariance = 2 * ey2

    b0 = BundleEKFContinuousTime("b0", 2)
    b0.f = fn.LinearFn(A0)
    b0.state.data["mu"] = mu0
    b0.dt = dt
    b0.state.data["mu"][1] = 2
    b0.state.data["Sigma"] = 2 * ey2

    m01 = MatcherEKF("m01", b0, b1)
    m01.print_state()


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

        for i in range(n_steps * 2):
            print_flush("Step {}/{} with brica".format(i, n_steps))
            s.step()

    visualize_bundle_rec(b0, y_rec)


if __name__ == '__main__':

    utils._print_level = 2 #5 is the noisiest, 1 is the most quiet

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
        print("===Starting UnitTest02===")
        print("-- A simple test of Bundle with Observer with no Matcher")
        plt.figure(2)
        test_bundle_and_observer(dt, n_steps, y_rec)
        plt.pause(0.2)

    if False:
        print("===Starting UnitTest03===")
        print("-- A simple test to link a Bundle, a Observer, and a Matcher")
        print("-- without brica")
        _with_brica = False
        plt.figure(4)
        test_MatcherEKF01(dt, n_steps, y_rec)
        plt.pause(0.2)

    if True:
        print("===Starting UnitTest04===")
        print("-- A simple test to link a Bundle, a Observer, and a Matcher")
        print("-- with brica")
        _with_brica = True
        plt.figure(3)
        test_MatcherEKF01(dt, n_steps, yrecs[0])
        plt.pause(0.2)

    plt.pause(5.0)
