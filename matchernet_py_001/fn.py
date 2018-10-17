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

def test_LinearFn():
    print("=== testLinearFn() ===")
    matA = np.array([[5,1,1],[1,2,3]],dtype=np.float32)
    x = np.array([[3,1]],dtype=np.float32)
    y = np.dot(x,matA)
    f = LinearFn(3,2)
    f.params["A"] = matA
    yout = f.value(x)
    print("matA:",matA)
    print("x:",x)
    print("y:",y)
    print("f(x):",yout)
    print("f_x(x):",f.x(x))
    print("f_xx(x):",f.xx(x))

def test_LinearFnXU():
    print("=== testLinearFnXU ===")
    matA = np.array([[5,1,1],[1,2,3]],dtype=np.float32)
    matB = np.array([[1,0,0]],dtype=np.float32)
    x = np.array([[3,1]],dtype=np.float32)
    u = np.array([[1]])
    y = np.dot(x,matA) + np.dot(u,matB)
    f = LinearFnXU(3,1,2)
    f.params["A"] = matA
    f.params["B"] = matB
    yout = f.value(x,u)
    print("matA:",matA)
    print("matB:",matB)
    print("x:",x)
    print("u:",u)
    print("y:",y)
    print("f(x,u):",yout)
    print("f_x(x,u):",f.x(x,u))
    print("f_xx(x,u):",f.xx(x,u))
    print("f_u(x,u):",f.u(x,u))
    print("f_xu(x,u):",f.xu(x,u))
    print("f_uu(x,u):",f.uu(x,u))

def test_all():
    test_LinearFn()
    test_LinearFnXU()

if __name__ == '__main__':
    test_all()
