"""
matchernet.py
=====

This module contains basic abstract classes that the BundleNet
architecture needs; 'Bundle' and 'Matcher'. It also includes
their tests , function 'test_abstract_bm'.
It imports BriCA2 (impremented with C++)
https://github.com/BriCA/BriCA2

"""

import brica
from brica import Component, VirtualTimeScheduler, Timing
import numpy as np
import utils
from utils import print1, print2, print3, print4, _print_level
import state
import time
import copy

#import pdb

zeros = utils.zeros

_with_brica = True
# A global flag to branch withBriCA2 and withoutBriCA versions

class Bundle(object):
    """
    'Bundle' is an abstract class that defines basic propaties of Bundles in a matchernet.

    Each bundle has
      a state
      a state transision dynamics
      arbitrary number of connections to Matchers
    """
    def __init__(self, name, initial_state_object):
        """ Create a new 'Bundle' instance.
        """
        self.name = name
        self.state = initial_state_object
        self.component = Component(self)
        self.component.make_out_port("state")

    def __call__(self, inputs):
        """ The main routine that is called from brica.
        """
        for key in inputs: # key is one of the matcher names
            if inputs[key] is not None:
                self.accept_feedback(inputs[key])

        self.update(inputs)
        return {"state": self.state}

    def accept_feedback(self, fb_state):
        """
        Accepting the feedback state 'fb_state' from one of the matchers
        which is linking to the current bundle.
        """
        print2("{} is accepting feedback".format(self.name))

    def print_state(self):
        """ Print the state of the current Bundle.
        Args: None.
        Returns: None.
        """
        print2("State of {n}".format(n=self.name))
        print2("self.state.data={x}".format(x=self.state.data))

    def update(self, inputs):
        """ Update the state of the current Bundle.
        This method should be override for any subclasses.
        """
        print3("{} is updating".format(self.name))

class Matcher(object):
    """
    'Matcher' is an abstract class that defines basic propaties of Matchers in a matchernet.
    Matcher is a component of matchernet.
    It connects Bundles, calculates energy, and returns a feedback state for each Bundle.
    Here, energy stands for a measure of contradiction among all the linking Bundles. The feedback stands for a signal that is used at the corresponding Bundle; the Bundle updates its state in order to decrease the energy.
    """
    def __init__(self, name, *bundles):
        self.name = name
        self.component = Component(self)
        self.results = {}
        for b in bundles:
            self.component.make_in_port(b.name)
            self.component.make_out_port(b.name)
            b.component.make_in_port(name)
            brica.connect(b.component, "state", self.component, b.name)
            brica.connect(self.component, b.name, b.component, name)
            print4( "{}".format(b.state.data))
            self.results[b.name] = copy.deepcopy(b.state)
        self.state = state.StatePlain(1) # Own state of the current matcher

    def __call__(self, inputs):
        """
        The main routine that is called from brica.
        The input variable  'inputs'  is a python dictionary object that brings all the current states of bundles.
        Ex.

          inputs = {"Bundle0", st0, "Bundle1", st1}

        The output variable 'results' is a python dictionary object which brings all the feedback states to the corresponding bundles.
        Ex.

          results = {"Bundle0", fbst0, "Bundle1", fbst1}

        """
        mu = {}
        for key in inputs:
            if inputs[key] is not None:
                self.accept_bundle_state(inputs[key])
        self.update(inputs)
        return self.results

    def accept_bundle_state(self, st):
        print2("Matcher {} is accepting_bundle_state".format(self.name))

    def update(self, inputs):
        self.print_state()

    def print_state(self):
        '''Prints the state of the self.'''
        print3("State of {n}".format(n=self.name))
        print3("self.state.data={x}".format(x=self.state.data))
        print3("self.results={x}".format(x=self.results))


# ====================
#   Test functions
# ====================

def test_abstract_bm():
    st = state.StateMuSigma(4)
    b0   = Bundle("Bundle0",st)
    b1   = Bundle("Bundle1",st)
    b2   = Bundle("Bundle2",st)

    m01 = Matcher("Matcher01",b0,b1)
    m02 = Matcher("Matcher02",b0,b2)
    m12 = Matcher("Matcher12",b1,b2)

    s = VirtualTimeScheduler()

    bt = Timing(0, 1, 1)
    bm = Timing(1, 1, 1)

    s.add_component( b0.component, bt)
    s.add_component( b1.component, bt)
    s.add_component( b2.component, bt)

    s.add_component( m01.component, bm)
    s.add_component( m02.component, bm)
    s.add_component( m12.component, bm)

    s.step()
    s.step()
    s.step()
    s.step()

    return s

if __name__ == '__main__':
    import time
    n=20
    start = time.time()
    # _print_level=1 # silent mode
    utils._print_level=3 # noisy mode
    s = test_abstract_bm()
    elapsed_time = time.time() - start
    print("abstract -- elapsed_time:{0}".format(elapsed_time) + "[sec]")
