# -*- coding: utf-8 -*-
import numpy as np

from matchernet.matchernet import Bundle, Matcher


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
