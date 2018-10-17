"""
matchernet.py
=====

This module contains a null demonstration of the BundleNet

"""
import brica
from brica import Component, VirtualTimeScheduler, Timing
import matchernet
from matchernet import Bundle, Matcher
import numpy as np
import utils
import state
import time
import copy
from operator import add
from functools import reduce


class BundleNull(Bundle):
    def __init__(self, name, n, delay=0):
        self.name = name
        # self.delay = delay
        self.delay = -1
        self.state = state.StatePlain(n)
        super(BundleNull, self).__init__(self.name, self.state)

    def accept_feedback(self, fb_state):
        self.state.data["mu"] += fb_state.data["mu"]

    def update(self, inputs):
        print("Updating {}".format(self.name))
        #if(self.delay):
            # ref = time.clock_gettime(time.CLOCK_MONOTONIC)
        #elapsed = 0
        #while elapsed < self.delay:
            # elapsed = time.clock_gettime(time.CLOCK_MONOTONIC) - ref

class MatcherNull2(Matcher):
    '''MatcherNull is a Matcher that connects two BundleNull s'''
    def __init__(self,name,b0,b1):
        self.name = name
        super(MatcherNull2, self).__init__(self.name, b0, b1)

    def update(self, inputs):
        mu = {}
        for key in inputs:
            mu[key] = inputs[key].data["mu"]
        mean = reduce(add, mu.values()) / len(inputs)
        for key in inputs:
            if inputs[key] is not None:
                self.results[key].data["mu"] = (mean - mu[key]) * 0.1



def test_null():
    b0   = BundleNull("Bundle0",4, 100)
    b1   = BundleNull("Bundle1",4, 200)
    b2   = BundleNull("Bundle2",4, 300)

    b0.state.data["mu"][0][1] = 1
    b1.state.data["mu"][0][2] = 10
    b2.state.data["mu"][0][3] = 100

    m01 = MatcherNull2("Matcher01",b0,b1)
    m02 = MatcherNull2("Matcher02",b0,b2)
    m12 = MatcherNull2("Matcher12",b1,b2)

    s = VirtualTimeScheduler()
    bt = Timing(0, 1, 1)
    bm = Timing(1, 1, 1)

    s.add_component(b0.component, bt)
    s.add_component(b1.component, bt)
    s.add_component(b2.component, bt)

    s.add_component( m01.component, bm)
    s.add_component( m02.component, bm)
    s.add_component( m12.component, bm)

    s.step()
    s.step()
    s.step()
    s.step()

    return s

if __name__ == '__main__':
    start = time.time()
    s = test_null()
    elapsed_time = time.time() - start
    print("null -- elapsed_time:{0}".format(elapsed_time) + "[sec]")
