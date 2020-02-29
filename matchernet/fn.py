"""
fn.py
=====

This module contains function handler classes that the BundleNet
architecture needs. It is overridden when you use chainer/TensorFlow to implement the arbitrary parametric functions.

"""

import autograd.numpy as np
from autograd import jacobian


class Fn(object):
    """An abstract class to implement numerical function
    that BundleNet uses.
    """

    def __init__(self, A):
        self.A = A
        self.x = jacobian(self.value, 0)

    def get_params(self):
        return self.A

    def value(self, x):
        """Numerically calculates f(x)
          x should be a numpy array of shape (dim_in, 1)
          outputs a numpy array of shape (dim_out, 1)
        """
        return 0


class LinearFn(Fn):
    """Linear function y = np.dot(A, x) and its derivatives.
    """

    def __init__(self, A):
        super(LinearFn, self).__init__(A)
        self.A = A
        self.x = jacobian(self.value, 0)

    def value(self, x):
        return self.A @ x


class LinearFnXU(object):
    """Linear function y = np.dot(A, x) + np.dot(B, u) and its derivatives.

    .. note:: The shapes of matrix A and matrix B must match

    """

    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.x = jacobian(self.value, 0)
        self.u = jacobian(self.value, 1)

    def value(self, x, u):
        return self.A @ x + self.B @ u
