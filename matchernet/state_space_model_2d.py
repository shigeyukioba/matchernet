import numpy as np

from matchernet import utils


class StateSpaceModel2Dim(object):
    def __init__(self, n_dim, A, g, sigma_w, sigma_z, x, y):
        if n_dim != 2:
            raise NotImplementedError
        self.n_dim = n_dim
        self.A = A
        self.g = g
        self.sigma_w = sigma_w
        self.sigma_z = sigma_z
        self.x = x
        self.y = y

    def simulation(self, n_steps, dt):
        F = utils.calc_matrix_F(self.A, dt)
        for i in range(n_steps):
            (x, y) = self._step(F)
            if i == 0:
                x_series = x
                y_series = y
            else:
                x_series = np.concatenate((x_series, x), axis=0)
                y_series = np.concatenate((y_series, y), axis=0)

        return x_series, y_series

    def _step(self, F):
        self.w = self.sigma_w * np.random.standard_normal(size=(1, self.n_dim))
        self.x = np.dot(self.x, F) + self.w
        self.z = self.sigma_z * np.random.standard_normal(size=(1, self.n_dim))
        self.y = self.x + self.z

        return self.x, self.y
