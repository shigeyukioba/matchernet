# -*- coding: utf-8 -*-
import numpy as np

from matchernet.matchernet import Bundle, Matcher
from matchernet import state
from matchernet import utils


class BundleEKFWithController(Bundle):
    def __init__(self, name, dt, f, Q, mu, Sigma, control_src_name):
        super(BundleEKFWithController, self).__init__(name)
        self.f = f
        self._initialize_control_params(dt)
        self._initialize_state(mu, Sigma)
        
        self.Q = Q
        self.control_src_name = control_src_name

        self.update_component()

    def __call__(self, inputs):
        # Update state
        for key in inputs:  # key is one of the matcher names
            if inputs[key] is not None:
                if key != self.control_src_name:
                    self.accept_feedback(inputs[key])

        # Step dynamics with applied control signal
        for key in inputs:  # key is one of the matcher names
            if inputs[key] is not None:
                if key == self.control_src_name:
                    u = inputs[key]["u"]
                    self.step_dynamics(u, self.dt)

        # Update timestamp
        self._countup()

        # Send state to Matcher
        results = {
            "mu": self.state.data["mu"],
            "Sigma": self.state.data["Sigma"],
            "time_stamp": self.time_stamp,
            "time_id": self.time_id
        }
        return {"state": results}

    def _initialize_control_params(self, dt):
        self.dt = dt
        self.time_stamp = 0.0
        self.time_id = 0

    def _countup(self):
        self.time_stamp += self.dt
        self.time_id += 1

    def _initialize_state(self, mu, Sigma):
        self.state = state.StateMuSigma(mu, Sigma)

    def accept_feedback(self, fbst):
        dmu = fbst["mu"]
        dSigma = fbst["Sigma"]
        mu = self.state.data["mu"]
        Sigma = self.state.data["Sigma"]

        self.state.data["mu"] = (mu + dmu).astype(np.float32)
        self.state.data["Sigma"] = (Sigma + dSigma).astype(np.float32)

    def step_dynamics(self, u, dt):
        mu = self.state.data["mu"]
        Sigma = self.state.data["Sigma"]
        Q = self.Q
        A = self.f.x(mu, u)
        F = utils.calc_matrix_F(A, dt)
        mu = mu + self.f.value(mu, u) * dt
        Sigma = dt * Q + F.T @ Sigma @ F
        Sigma = utils.regularize_cov_matrix(Sigma)

        self.state.data["mu"] = mu
        self.state.data["Sigma"] = Sigma
        # ["time_stamp"] is updated in the method self._countup()


# TODO: マージ後に置き換え
class MatcherEKF(Matcher):
    def __init__(self, name, b0, b1, g0, g1):
        self.name = name
        super(MatcherEKF, self).__init__(name, b0, b1)
        self.b0name = b0.name
        self.b1name = b1.name
        self.g0 = g0
        self.g1 = g1
        self._initialize_model()
        self.update_component()

    def _initialize_model(self):
        self.lnL = 0
        self.err2 = 0

    def __call__(self, inputs):
        self.update(inputs)
        return self.results

    def forward(self):
        self.lnL_t = 0
        z = self.g0.value(self.mu0) - self.g1.value(self.mu1)
        C0 = self.g0.x(self.mu0)
        C1 = self.g1.x(self.mu1)
        S = C0 @ self.Sigma0 @ C0.T + C1 @ self.Sigma1 @ C1.T
        SI = np.linalg.inv(S)
        dum_sign, logdet = np.linalg.slogdet(S)
        self.lnL_t -= z @ SI @ z.T / 2.0
        self.err2 += z @ z.T
        self.lnL_t -= logdet / 2.0
        K0 = self.Sigma0 @ C0.T @ SI
        K1 = self.Sigma1 @ C1.T @ SI

        self.dmu0 = -K0 @ z
        self.dmu1 = K1 @ z
        self.dSigma0 = -K0 @ C0 @ self.Sigma0
        self.dSigma1 = -K1 @ C1 @ self.Sigma1
        self.lnL += self.lnL_t

    def backward(self):
        pass

    def update(self, inputs):
        self.b0state, self.b1state = inputs[self.b0name], inputs[self.b1name]
        d0, d1 = self.b0state, self.b1state

        self.mu0 = d0["mu"]
        self.Sigma0 = d0["Sigma"]
        #self.ts0 = d0["time_stamp"]
        self.mu1 = d1["mu"]
        self.Sigma1 = d1["Sigma"]
        #self.ts1 = d1["time_stamp"]

        #self.ts0_recent = self.ts0
        #self.ts1_recent = self.ts1

        self.forward()
        self.backward()

        self.results[self.b0name]["mu"] = self.dmu0.astype(np.float32)
        self.results[self.b0name]["Sigma"] = self.dSigma0.astype(np.float32)
        self.results[self.b1name]["mu"] = self.dmu1.astype(np.float32)
        self.results[self.b1name]["Sigma"] = self.dSigma1.astype(np.float32)
