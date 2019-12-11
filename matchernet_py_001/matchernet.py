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
import copy

from matchernet_py_001 import state
from matchernet_py_001 import utils
from matchernet_py_001.utils import print2, print3, print4

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
        Args:
            None.
        Returns:
            None.
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

    Args:
        name (str): matcher name.
        *bundles (:class:`~chainer.Variable`):
    """
    def __init__(self, name, *bundles):
        self.name = name
        self.state = state.StatePlain(1) # Own state of the current matcher
        self.results = {}
        self.bundles = bundles
        for bundle in self.bundles:
            self.results[bundle.name] = copy.deepcopy(bundle.state)
        self.component = None
        self.update_component()

    def update_component(self):
        self.component = Component(self)
        for bundle in self.bundles:
            self.component.make_in_port(bundle.name)
            self.component.make_out_port(bundle.name)
            bundle.component.make_in_port(self.name)
            brica.connect(bundle.component, "state", self.component, bundle.name)
            brica.connect(self.component, bundle.name, bundle.component, self.name)
            print4( "{}".format(bundle.state.data))

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
        """Prints the state of the self.
        """
        print3("State of {n}".format(n=self.name))
        print3("self.state.data={x}".format(x=self.state.data))
        print3("self.results={x}".format(x=self.results))
