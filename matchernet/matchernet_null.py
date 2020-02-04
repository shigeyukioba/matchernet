"""
matchernet.py
=====

This module contains a null demonstration of the BundleNet

"""
import logging
from log import logging_conf
from operator import add
from functools import reduce

from matchernet.matchernet import Bundle, Matcher
from matchernet import state

logging_conf.set_logger_config("./log/logging.json")
logger = logging.getLogger(__name__)


class BundleNull(Bundle):
    def __init__(self, name, mu, delay=0, logger=logger):
        self.logger = logger.getChild(self.__class__.__name__)
        self.name = name
        # self.delay = delay
        self.delay = -1
        self.state = state.StatePlain(mu)
        super(BundleNull, self).__init__(self.name)

    def accept_feedback(self, fb_state):
        self.state.data["mu"] += fb_state["mu"]

    def __call__(self, inputs):
        for key in inputs:
            if inputs[key] is not None:
                self.accept_feedback(inputs[key])

        self.update(inputs)
        results = {}
        results["mu"] = self.state.data["mu"]
        return {
            "state" : results
        }

    def update(self, inputs):
        self.logger.debug("Updating {}".format(self.name))
        # if(self.delay):
        # ref = time.clock_gettime(time.CLOCK_MONOTONIC)
        # elapsed = 0
        # while elapsed < self.delay:
        # elapsed = time.clock_gettime(time.CLOCK_MONOTONIC) - ref


class MatcherNull2(Matcher):
    """MatcherNull is a Matcher that connects two BundleNull s
    """

    def __init__(self, name, b0, b1):
        self.name = name
        super(MatcherNull2, self).__init__(self.name, b0, b1)

    def update(self, inputs):
        mu = {}
        for key in inputs:
            mu[key] = inputs[key]["mu"]
        mean = reduce(add, mu.values()) / len(inputs)
        for key in inputs:
            if inputs[key] is not None:
                self.results[key]["mu"] = (mean - mu[key]) * 0.1
