# -*- coding: utf-8 -*-
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
from brica import Component


class Bundle(object):
    """
    'Bundle' is an abstract class that defines basic propaties of Bundles in a matchernet.

    Each bundle has
      a state
      a state transision dynamics
      arbitrary number of connections to Matchers
    """
    def __init__(self, name):
        """ Create a new 'Bundle' instance.
        """
        self.name = name

    def update_component(self):
        self.component = Component(self)
        self.component.make_out_port("state")

    def __call__(self, inputs):
        return {}


class Matcher(object):
    """
    'Matcher' is an abstract class that defines basic propaties of Matchers in a matchernet.
    Matcher is a component of matchernet.
    It connects Bundles, calculates energy, and returns a feedback state for each Bundle.
    Here, energy stands for a measure of contradiction among all the linking Bundles. 
    The feedback stands for a signal that is used at the corresponding Bundle; 
    the Bundle updates its state in order to decrease the energy.
    """
    def __init__(self, name, *bundles):
        self.name = name
        self.bundles = bundles
        self.update_component()

    def update_component(self):
        component = Component(self)
        
        for b in self.bundles:
            component.make_in_port(b.name)
            component.make_out_port(b.name)
            b.component.make_in_port(self.name)
            
            brica.connect(b.component, "state", component, b.name)
            brica.connect(component, b.name, b.component, self.name)
            
        self.component = component

    def __call__(self, inputs):
        return {}
