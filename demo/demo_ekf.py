import sys
import os
sys.path.append(os.getcwd())

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
def visualize_bundle_rec(b, yrec=None):

    murec = b.record["mu"]
    sigmarec = b.record["diagSigma"]
    time_stamp = b.record["time_stamp"]
    numofTimesteps = murec.shape[0]
    timestamp = np.array(range(0,numofTimesteps))

    print3("murec.shape={}".format(murec.shape))
    print3("murec.dtype={}".format(murec.dtype))
    print3("sigmarec.shape={}".format(sigmarec.shape))
    print3("sigmarec.dtype={}".format(sigmarec.dtype))
    print3("timestamp.shape={}".format(timestamp.shape))
    print4("murec={}".format(murec))
    print4("sigmarec={}".format(sigmarec))
    print4("timestamp={}".format(timestamp))

    plt.subplot(221)
    plt.plot(murec[:,0],murec[:,1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory")


    plt.subplot(222)
    yd = murec[:,0]-sigmarec[:,0]
    yu = murec[:,0]+sigmarec[:,0]
    print4("sigmarec[:,0]={}".format(sigmarec[:,0]))
    print4("murec[:,0]={}".format(murec[:,0]))
    print3("murec[:,0].shape={}".format(murec[:,0].shape))
    print3("sigmarec[:,0].shape={}".format(sigmarec[:,0].shape))
    print3("murec[:,1].shape={}".format(murec[:,1].shape))
    print3("sigmarec[:,1].shape={}".format(sigmarec[:,1].shape))
    print3("yu.shape={}".format(yu.shape))
    plt.fill_between(time_stamp,yd,yu,facecolor='y', alpha=0.5)
    plt.plot(time_stamp,murec[:,0])
    plt.ylabel("X")
    plt.xlabel("time")

    plt.subplot(224)
    yd = murec[:,1]-sigmarec[:,1]
    yu = murec[:,1]+sigmarec[:,1]
    plt.fill_between(time_stamp,yd,yu,facecolor='y', alpha=0.5)
    plt.plot(time_stamp,murec[:,1])
    #plt.scatter(range(0,5000-1),murec[:,0])
    plt.ylabel("Y")
    plt.xlabel("time")

    if yrec is None:
        return
    else:
        plt.subplot(221)
        plt.scatter(yrec[:,0], yrec[:,1],s=2)
        plt.subplot(222)
        plt.scatter(time_stamp, yrec[:,0],s=2)
        plt.subplot(224)
        plt.scatter(time_stamp, yrec[:,1],s=2)


#=======================================================================
#  Test functions for the matchernet_ekf.py
#=======================================================================

mu0 = np.array([[0,1.0]],dtype=np.float32)
A0 = np.array([[-0.1,2],[-2,-0.1]],dtype=np.float32)
ey2 = np.eye(2,dtype=np.float32)

def test_BundleEKFContinuousTime01(dt, numSteps):
    # test of BundleEKFContinuousTime with a two dimensional linear dynamics
    # dx = dt * F * x + Q * dw
    b = BundleEKFContinuousTime("B0",2)
    b.dt = dt
    b.print_state()
    b.f=fn.LinearFn(2,2)
    b.f.params["A"] = A0
    b.state.data["mu"] = mu0

    dummy_input = {} # list of matchers (#matcher=0)
    for i in range(numSteps):
        b(dummy_input)

    visualize_bundle_rec(b)


def test_bundle_and_observer(dt, numSteps, yrec):
    b0 = observer.Observer("b0",yrec)
    b1 = BundleEKFContinuousTime("b1",2)
    b1.f.params["A"] = A0
    b1.state.data["mu"] = mu0
    b1.dt = dt
    dummy_input = {} # list of matchers (#matcher=0)
    for i in range(numSteps):
        b0(dummy_input)
        b1(dummy_input)
        y=b0.get_state()
        if i==0:
            yrec2=y
        else:
            yrec2 = np.concatenate((yrec2,y),axis=0)

    timestamp=np.array(range(0,numSteps))
    visualize_bundle_rec(b1,yrec2)


def test_MatcherEKF01(dt,numSteps,yrec):
    '''
    Run a matchernet of the following structure
    b0 --- m01 --- b1
    '''

    b1 = observer.Observer("b1",yrec)
    b1.obs_noise_covariance = 2 * ey2

    b0 = BundleEKFContinuousTime("b0",2)
    b0.f.params["A"] = A0
    b0.state.data["mu"] = mu0
    b0.dt = dt
    b0.state.data["mu"][0][1]=2
    b0.state.data["Sigma"] = 2 * ey2

    m01 = MatcherEKF("m01",b0,b1)
    m01.print_state()


    if _with_brica is False:
        for i in range(numSteps):
            print_flush("Step {}/{} with brica".format(i,numSteps))
            inputs_to_m01 = {"b0":b0.state, "b1":b1.state}
            results = m01(inputs_to_m01)
            inputs_to_b0 = {"m01":results["b0"]}
            s0 = b0(inputs_to_b0)
            inputs_to_b1 = {"m01":results["b1"]}
            s1 = b1(inputs_to_b1)
    else:
        s = VirtualTimeScheduler()

        bt = Timing(0, 1, 1)
        bm = Timing(1, 1, 1)

        s.add_component( b0.component, bt)
        s.add_component( b1.component, bt)
        s.add_component( m01.component, bm)

        for i in range(numSteps*2):
            print_flush("Step {}/{} with brica".format(i,numSteps))
            s.step()

    visualize_bundle_rec(b0, yrec)


if __name__ == '__main__':

    utils._print_level = 2 #5 is the noisiest, 1 is the most quiet

    if False:
        print("===Starting UnitTest01===")
        print("-- A simple test of Bundle with no Matcher")
        plt.figure(1)
        test_BundleEKFContinuousTime01(dt, numSteps)
        plt.pause(0.2)

    dt = 0.02
    numSteps = 500

    # preparing a list of three simulation sequences
    yrecs = []
    for i in range(1):
        sm=StateSpaceModel2Dim(dt)
        sm.A = A0
        sm.x = mu0[0]
        sm.dt = dt
        (xrec,yrec)=sm.simulation(dt, numSteps)
        yrecs.append(yrec)

    plt.figure(0)
    for i in range(1):
        plt.subplot(2,2,i+1)
        plt.scatter(yrecs[i][:,0],yrecs[i][:,1],s=2)
        plt.title("sequence {}".format(i))
    plt.pause(1)

    if False:
        print("===Starting UnitTest02===")
        print("-- A simple test of Bundle with Observer with no Matcher")
        plt.figure(2)
        test_bundle_and_observer(dt, numSteps, yrec)
        plt.pause(0.2)

    if False:
        print("===Starting UnitTest03===")
        print("-- A simple test to link a Bundle, a Observer, and a Matcher")
        print("-- without brica")
        _with_brica = False
        plt.figure(4)
        test_MatcherEKF01(dt, numSteps, yrec)
        plt.pause(0.2)

    if True:
        print("===Starting UnitTest04===")
        print("-- A simple test to link a Bundle, a Observer, and a Matcher")
        print("-- with brica")
        _with_brica = True
        plt.figure(3)
        test_MatcherEKF01(dt, numSteps, yrecs[0])
        plt.pause(0.2)


    plt.pause(5.0)
