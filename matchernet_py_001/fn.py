"""
fn.py
=====

This module contains function handler classes that the BundleNet
architecture needs. It is overridden when you use chainer/TensorFlow to implement the arbitrary parametric functions.

"""

import numpy as np

class Fn(object):
    ''' An abstract class to implement numerical function
    that BundleNet uses.
    '''
    def __init__(self, dim_in, dim_out):
        self.shape =(dim_in,dim_out)
        self.params = {"none":0}

    def get_params():
        return self.params

    def value(self, x):
        ''' Numerically calculates f(x)
          x should be a numpy array of shape (dim_in, 1)
          outputs a numpy array of shape (dim_out, 1)
         '''
        return 0
    def x(self, x):
        ''' Numerically calculates df/dx
          outputs a numpy array of shape (dim_out, dim_in)
         '''
        return 0
    def xx(self, x):
        ''' Numerically calculates df^2/dx^2
          outputs a numpy array of shape (dim_out, dim_in, dim_in)
         '''
        return 0

class LinearFn(Fn):
    ''' Linear function y = np.dot( x, A ) and its derivatives.'''
    def __init__(self, dim_in, dim_out):
        super(LinearFn, self).__init__(dim_in, dim_out)
        A = np.zeros((dim_out,dim_in),dtype=np.float32)
        self.params = {"A":A}

    def value(self, x):
        return np.dot( x, self.params["A"] )

    def x(self, x):
        return self.params["A"]

    def xx(self, x):
        return np.zeros(self.shape,dtype=np.float32)

class LinearFnXU(Fn):
    ''' Linear function y = np.dot( x, A ) + np.dot( u, B )
     and its derivatives.'''
    def __init__(self, dim_x, dim_u, dim_out):
        super(LinearFnXU, self).__init__(dim_x, dim_out)
        A = np.zeros((dim_out,dim_x),dtype=np.float32)
        B = np.zeros((dim_out,dim_u),dtype=np.float32)
        self.params = {"A":A, "B":B}
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dim_out = dim_out

    def value(self, x, u):
        return np.dot( x, self.params["A"] ) + np.dot( u, self.params["B"])

    def x(self, x, u):
        return self.params["A"]

    def u(self, x, u):
        return self.params["B"]

    def xx(self, x, u):
        return np.zeros((self.dim_out,self.dim_x,self.dim_x),dtype=np.float32)

    def xu(self, x, u):
        return np.zeros((self.dim_out,self.dim_x,self.dim_u),dtype=np.float32)

    def uu(self, x, u):
        return np.zeros((self.dim_out,self.dim_u,self.dim_u),dtype=np.float32)
