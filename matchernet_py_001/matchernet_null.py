"""
matchernet.py
=====

This module contains a null demonstration of the BundleNet

"""
from .matchernet import Bundle, Matcher
from matchernet_py_001 import state
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
    """MatcherNull is a Matcher that connects two BundleNull s
    """
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
