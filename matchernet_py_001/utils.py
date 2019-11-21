# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
import sys

def print_flush(s):
    sys.stdout.write("\r"+s)
    sys.stdout.flush()

# print variants for debug
_print_level = 1

def print5(fm):
    if _print_level>5:
        print(fm)

def print4(fm):
    if _print_level>4:
        print(fm)

def print3(fm):
    if _print_level>3:
        print(fm)

def print2(fm):
    if _print_level>2:
        print(fm)

def print1(fm):
    if _print_level>1:
        print(fm)

def zeros(size):
    return np.zeros(size).astype(np.float32)

def regularize_cov_matrix(a, mineig=1e-5):
    """
    regularize the covariance matrix  a  so that its minimum eigen value is larger than  mineig.
    """
    l, p = np.linalg.eigh(a)
      # Note: using eigh() rather than eig()
    n = l.size
    for i in range(n):
        if l[i] < mineig:
            l[i] = mineig
    ar = np.dot(np.dot(p, np.diag(l)), p.T)
    return ar

def q_plot001(n, q_array, x_array):
    trange = range(0, q_array.shape[0])
    for i in range(n):
        plt.subplot(n, 1, i+1)
        mu = q_array[:,i+1]
        s = np.sqrt( q_array[:,1+n+(n*i+i)] )
        x = x_array[:,i]
        q_plot002(trange, mu, s, color='green')
        plt.plot(x, 'b.-')
        plt.ylabel("x{i}".format(i=i))
    plt.xlabel("Time")

def q_plot002(trange, mu, s, color='blue'):
    plt.fill_between(trange, mu-s, mu+s,
                        edgecolor='none', facecolor=color,alpha=0.2)
    plt.plot(trange, mu, '-', color=color)

def calc_matrix_F(A, dt):
    n = A.shape[1]
    A2 = np.dot(A, A)
    A3 = np.dot(A2, A)
    F=np.identity(n, dtype=np.float32) + dt * A + dt*dt*A2/2.0 + dt**3 * A3/6.0
    return F
