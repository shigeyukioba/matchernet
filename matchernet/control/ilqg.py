# -*- coding: utf-8 -*-
import numpy as np


class iLQG:
    def __init__(self, dynamics, cost):
        self.dynamics = dynamics
        self.cost = cost

    def optimize(self,
                 x_init,
                 u_init,
                 T,
                 iter_max=20,
                 stop_criteria=1e-3):

        assert x_init.shape == (self.dynamics.x_dim,)
        assert u_init.shape == (T, self.dynamics.u_dim)
        
        x_list = self.init_trajectory(x_init, u_init, T)
        
        u_list = np.copy(u_init)
        
        for i in range(iter_max):
            print("iter={}".format(i))
            k_list, K_list = self.backward(x_list, u_list, T)
            x_list, u_list, diff = self.forward(x_list, u_list, x_init,
                                                k_list, K_list, T)

            if(diff < stop_criteria):
                print("it={}, diff={}".format(i, diff))
                break
            
        return x_list, u_list, K_list
    
    def init_trajectory(self, x_init, u_init, T):
        x_list = [x_init]
        
        for i in range(T):
            next_x = self.dynamics.value(x_list[i], u_init[i])
            x_list.append(next_x)
            
        return x_list

    def forward(self, x_list, u_list, x_init, k_list, K_list, T):
        next_x_list = [x_init]
        next_u_list = []
        
        diff = 0.0
        alpha = 1.0
        
        for t in range(T):
            next_u = u_list[t] + alpha * k_list[t] + \
                     K_list[t] @ (next_x_list[t] - x_list[t])
            next_u_list.append(next_u)
            next_x = self.dynamics.value(next_x_list[t], next_u_list[t])
            next_x_list.append(next_x)
            
            diff += np.sum(np.abs(u_list[t] - next_u_list[t]))

        return next_x_list, next_u_list, diff

    def backward(self, x_list, u_list, T):
        k_list = []
        K_list = []

        u_zero = np.zeros_like(u_list[0])

        # Derivatives of the terminal cost
        lx  = self.cost.x( x_list[T], u_zero, T)
        lxx = self.cost.xx(x_list[T], u_zero, T)
        
        Vx  = lx  # (x_dim,)
        Vxx = lxx # (x_dim,x_dim)
        
        assert Vx.shape  == (self.dynamics.x_dim,)
        assert Vxx.shape == (self.dynamics.x_dim, self.dynamics.x_dim)
        
        for t in range(T-1, -1, -1):
            x = x_list[t]
            u = u_list[t]
            
            assert x.shape == (self.dynamics.x_dim,)
            assert u.shape == (self.dynamics.u_dim,)
            
            # Derivatives of the dynamics
            fx = self.dynamics.x(x, u)
            fu = self.dynamics.u(x, u)

            # Derivatives of the cost
            lx  = self.cost.x( x, u, t)
            lxx = self.cost.xx(x, u, t)
            lu  = self.cost.u( x, u, t)
            luu = self.cost.uu(x, u, t)
            lux = self.cost.ux(x, u, t)

            # Derivatives of the Q function
            Qx  = lx  + fx.T @ Vx
            Qu  = lu  + fu.T @ Vx
            Qxx = lxx + fx.T @ Vxx @ fx
            Quu = luu + fu.T @ Vxx @ fu
            Qux = lux + fu.T @ Vxx @ fx

            # Regularlize
            # TODO: add adaptive adjustment for lambd
            #lambd = 0.1
            #lambd = 0.5
            lambd = 0.3
            Quu = Quu + np.eye(self.dynamics.u_dim) + lambd
            
            assert Qx.shape  == (self.dynamics.x_dim,)
            assert Qu.shape  == (self.dynamics.u_dim,)
            assert Qxx.shape == (self.dynamics.x_dim, self.dynamics.x_dim)
            assert Quu.shape == (self.dynamics.u_dim, self.dynamics.u_dim)
            assert Qux.shape == (self.dynamics.u_dim, self.dynamics.x_dim)
            
            Quu_inv = np.linalg.inv(Quu)
            
            k = -Quu_inv @ Qu
            K = -Quu_inv @ Qux
            
            assert k.shape == (self.dynamics.u_dim,)
            assert K.shape == (self.dynamics.u_dim, self.dynamics.x_dim)

            Vx  = Qx  + K.T @ Quu @ k + K.T @ Qu  + Qux.T @ k
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            
            assert Vx.shape  == (self.dynamics.x_dim,)
            assert Vxx.shape == (self.dynamics.x_dim, self.dynamics.x_dim)
            
            k_list.append(k)
            K_list.append(K)
            
        k_list.reverse()
        K_list.reverse()

        return k_list, K_list
