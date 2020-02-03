import logging
from log import logging_conf
from operator import add
from functools import reduce
import numpy as np

import brica
from brica import Component, VirtualTimeScheduler, Timing
from matchernet import state
import copy

logging_conf.set_logger_config("./log/logging.json")
logger = logging.getLogger(__name__)


class NullBundle(object):
    def __init__(self, name, mu, Sigma, logger=logger):
        super(NullBundle, self).__init__()
        self.logger = logger.getChild(self.__class__.__name__)
        self.name = name

        self.state = state.StateMuSigma(mu, Sigma)
        self.component = Component(self)
        self.component.make_out_port("state")

    def __call__(self, inputs):
        for key in inputs:
            if inputs[key] is not None:
                self.state.data["mu"] += inputs[key]["mu"]
        logger.debug("{} state: {}".format(self.name, self.state.data["mu"]))
        return {"state" : self.state.data}


class NullMatcher(object):
    def __init__(self, name, *bundles):
        super(NullMatcher, self).__init__()
        self.name = name
        self.result_state = {}

        for bundle in bundles:
            self.result_state[bundle.name] = {}

        self.component = Component(self)
        for bundle in bundles:
            self.component.make_in_port(bundle.name)
            self.component.make_out_port(bundle.name)

            bundle.component.make_in_port(name)

            brica.connect(bundle.component, "state", self.component, bundle.name)
            brica.connect(self.component, bundle.name, bundle.component, name)

    def __call__(self, inputs):
        mu = {}
        for key in inputs:
            if inputs[key] is not None:
                mu[key] = inputs[key]["mu"]
        mean = reduce(add, mu.values()) / len(inputs)
        for key in inputs:
            if inputs[key] is not None:
                self.result_state[key]["mu"] = (mean - mu[key]) * 0.1
        return self.result_state


def main():
    Sigma = np.eye(4, dtype=np.float32)

    b0 = NullBundle("Bundle0", mu=np.array([0, 1, 0, 0]).astype(np.float32), Sigma=Sigma)
    b1 = NullBundle("Bundle1", mu=np.array([0, 0, 10, 0]).astype(np.float32), Sigma=Sigma)
    b2 = NullBundle("Bundle2", mu=np.array([0, 0, 0, 100]).astype(np.float32), Sigma=Sigma)

    m01 = NullMatcher("Matcher01", b0, b1)
    m02 = NullMatcher("Matcher02", b0, b2)
    m12 = NullMatcher("Matcher12", b1, b2)

    s = VirtualTimeScheduler()
    bt = Timing(0, 1, 1)
    bm = Timing(1, 1, 1)

    s.add_component(b0.component, bt)
    s.add_component(b1.component, bt)
    s.add_component(b2.component, bt)

    s.add_component(m01.component, bm)
    s.add_component(m02.component, bm)
    s.add_component(m12.component, bm)

    s.step()
    s.step()
    s.step()
    s.step()


if __name__ == '__main__':
    main()
