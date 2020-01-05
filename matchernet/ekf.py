import logging
from log import logging_conf
import numpy as np

from matchernet import fn
from matchernet.matchernet import Bundle, Matcher
from matchernet import state
from matchernet import utils

logging_conf.set_logger_config("./log/logging.json")
logger = logging.getLogger(__name__)


class BundleEKFContinuousTime(Bundle):
    """Class BundleEKFContinuousTime is a Bundle part of an extended Kalman filter (EKF) model implemented as the BundleNet.

    See matchernet.py for general concept of the BundleNet, especially the general relationship between the "Bundle" and the "Matcher".

    This class "BundleEKFContinuousTime" manages an internal state  x  and its dynamics model  f.
    The dynamics is defined as a probabilistic differential equation,
       dx/dt = f(x) + Q dw,
    where  x  is an  n  dimensional state vector variable,
           f  determines the intrinsic non-linear dynamics
                 by a map from n-dim to n-dim,
           dw is a Wiener process.
    This dynamics is locally linearlized and temporarily descritized
    as followings,
       x(t+dt) = dot( x(t), F ) + w(t), w(t) ~ N(0, dt*Q )
    where  F = exp( dt*A )  is the linear coefficient matrix
    that is defined with the Jacobian matrix  A = df/dx.

    This class stores the internal state as a temporal estimation of
    the posterior probability density function
       q(x) = N( mu, Sigma )
    that is parameterized as
           mu = self.state["mu"]  (1,n)-np.array
    and
           Sigma = self.state["Sigma"]  (n,n)-np.array.

    Dynamics function  f  is stored as a function instance
        self.f
    of a class  Fn   which is defined in  fn.py .


    In EKF algorithm, the BundleEKFContinuousTime updates
       posterior  q(x)  through following steps.

       1. Updates posterior  q(x)  according to the intrinsic dynamics
       with an arbitrary time step  dt.
       It is implemented in the method  self.step_dynamics().

       2. Updates posterior  q(x)  in order to increase objective functions
       that are defined in Matchers.
       It is implemented in the method  self.accept_feedback().


    One Bundle can connect to multiple Matchers. Thus, the difference signals
    from all the connected Matchers are applied simultaneously.

    The dynamics function  f  is fixed at the initialized value.
    See an extented class
        BundleEKFContinuousTimeTrainable (to be implemented soon)
    for a similar class with trainable dynamics function.

    """

    def __init__(self, name, n, f, logger=logger):
        self.logger = logger.getChild(self.__class__.__name__)
        self.n = n  # Dimsnsionarity of the state variable
        self.name = name
        self.state = state.StateMuSigma(n)
        self._initialize_control_params()
        self._initialize_state(n)
        self.f = f
        # self.bw = matchernet.bundleWeight(numSteps)
        self.record = {}
        self._first_call_of_state_record = True
        super(BundleEKFContinuousTime, self).__init__(self.name, self.state)

    def __call__(self, inputs):
        """The main routine that is called from brica.
        """
        for key in inputs:  # key is one of the matcher names
            if inputs[key] is not None:
                self.logger.debug("accepting feedback from {}".format(key))
                self.accept_feedback(inputs[key])
        self.step_dynamics(self.dt)
        self._countup()
        self.state.data["Sigma"] = utils.regularize_cov_matrix(self.state.data["Sigma"])
        self._state_record()
        self.logger.debug("mu={}".format(self.state.data["mu"]))

        return {"state": self.state}

    def _initialize_control_params(self):
        self.id = 0
        self.dt = 0.01
        self.callcount = 0
        self.is_optimizer_ready = False
        self.lr = 0.0001  # Leaning rate for dynamics  f

    def _countup(self):
        self.id = self.id + 1
        self.state.data["time_stamp"] = self.state.data["time_stamp"] + self.dt
        self.callcount = self.callcount + 1

    def _state_record(self):
        mu = np.array(self.state.data["mu"], dtype=np.float32)
        sigma = np.array([np.diag(self.state.data["Sigma"])], dtype=np.float32)
        ts = np.array([self.state.data["time_stamp"]], dtype=np.float32)

        if self._first_call_of_state_record:
            self.record = {"mu": mu, "diagSigma": sigma, "time_stamp": ts}
            self._first_call_of_state_record = False
        else:
            self.record["mu"] = np.vstack((self.record["mu"], mu))
            self.record["diagSigma"] = np.concatenate((self.record["diagSigma"], sigma), axis=0)
            self.record["time_stamp"] = np.concatenate((self.record["time_stamp"], ts), axis=0)

    def _initialize_state(self, n):
        self.state.data["id"] = self.id
        self.state.data["mu"] = utils.zeros(n)
        self.state.data["Sigma"] = 1.0 * np.identity(self.n, dtype=np.float32)
        self.state.data["Q"] = 1.0 * np.identity(self.n)

    def accept_feedback(self, fbst):
        """Overriding matchernet.Bundle.accept_feedback()
        This method updates the state of the current Bundle with accepting a feedback, fbst, from each Matcher linking from the current Bundle, according to the following update rule.

            mu <-- mu + dmu,
            Sigma <-- Sigma - dSigma,

        where  mu  and  Sigma  stands for the probabilistic state of the current bundle  q(x) = N( mu, Sigma ),
        and  dmu  and  dSigma  stands for the feedback state coming from the corresponding Matcher.
        """
        dmu = fbst.data["mu"]
        dSigma = fbst.data["Sigma"]
        mu = self.state.data["mu"]
        Sigma = self.state.data["Sigma"]
        # Q = self.state.data["Q"]

        self.logger.debug("dmu={}".format(dmu))
        self.logger.debug("dSigma={}".format(dSigma))
        weight = 1.0

        self.state.data["mu"] = (mu + weight * dmu).astype(np.float32)
        self.state.data["Sigma"] = (Sigma + weight * dSigma).astype(np.float32)
        # self.Q = (1-weight*self.lr) * Q + weight*self.lr * np.dot(dmu.T,dmu)

    def step_dynamics(self, dt):
        """This method updates  self.state  using the dynamics model,

            dx/dt  =  f(x) + Q dw.

        In order to (1) temporally discretization, (2) locally liniarization, and (3) consideration of probabilistic state, the actual update rule becomes the following manner

            mu <-- mu + dot( x(t), F )
            Sigma <-- F' * Sigma * F + dt * Q

        where  F = exp( dt*A )  is a linear coefficient matrix calculated with the Jacobian matrix  A = (df/dx).
        """
        mu = self.state.data["mu"]
        Sigma = self.state.data["Sigma"]
        Q = self.state.data["Q"]
        A = self.f.dx(mu)
        # Note:  mu.shape = (n, ), A.shape = (n,n)
        F = utils.calc_matrix_F(A, dt)
        mu = np.dot(F, mu)
        Sigma = dt * Q + np.dot(np.dot(F.T, Sigma), F)
        self.state.data["mu"] = mu
        self.state.data["Sigma"] = Sigma
        # ["time_stamp"] is updated in the method self._countup()


class MatcherEKF(Matcher):
    """Class MatcherEKF is a Matcher part of an extended Kalman filter (EKF) model implemented as the BundleNet.

    See matchernet.py for general concept of the BundleNet, especially the general relationship between the "Bundle" and the "Matcher".

    In short, the followings are what MatcherEKF does in the BundleNet.
     (1) Defines a link between two bundles  b0  and  b1
     (2) Receives current states of bundles  b0  and  b1, which are probability density function  q(x0) = N( mu0, Sigma0 )  and  q(x1) = N( mu1, Sigma1 )
     (3) Evaluates error  E  (or log likelihood  lnL  )  defined as a function of the current bundle states.
     (4) Calculates the derivatives of  E  w.r.t. the states,
         dmu0 = d E / d mu0, dSigma0 = d E / d Sigma0
         dmu1 = d E / d mu1, dSigma1 = d E / d Sigma1
     (5) Returns the derivatives as "feedback state" from the current Matcher to the corresponding Bundles.

    The MatcherEKF defines a error (or contradiction) between the two bundles as

       E =  z S^{-1} z'

    and corresponding log-likelihood function as

       lnl = - (1/2) z S^{-1} z' - (1/2) ln det 2 pi S

    where

       z = g0( mu0 ) - g1( mu1 )

    stands for the error (or contradiction) between the states of the two Bundles.  g0()  and  g1()  are arbitrary user-defined functions. The user can set  g0  and  g1  identity function  mu0 = g0(mu0)  and  mu1 = g1(mu1)  for the simplest case.
    State variables of the bundles are denoted as normal distributions

       q(x0)  ==  N( mu0,  Sigma0 )
       q(x1)  ==  N( mu1,  Sigma1 )

    Then, the total variance  S  of the current matcher is defined as

       S = C0 Sigma0 C0' + C1 Sigma1 C1'

    where  C0 = (dg0/dx)  and  C1 = (dg1/dx)  are Jacobian matrices. Note that C0 and C1 are identity matrices and  S = Sigma0 + Sigma1  holds in the simplest case.
    """

    def __init__(self, name, b0, b1, logger=logger):
        self.logger = logger.getChild(self.__class__.__name__)
        self.name = name
        super(MatcherEKF, self).__init__(name, b0, b1)
        self.b0name = b0.name
        self.b1name = b1.name
        self.n0 = b0.state.n  # dim. of B0
        self.n1 = b1.state.n  # dim. of B1
        self._first_call_of_state_record = True
        self._initialize_model()
        self.ts0_recent = -1  # the most recent value of the time_stamp of b0
        self.ts1_recent = -1  # that of b1
        self.update_component()

    def _initialize_model(self):
        self.n = self.n0  # dim. of g0(x0), g1(x1)
        # self.g0 is an identity function as a default observation model
        self.g0 = fn.LinearFn(np.eye(self.n0, dtype=np.float32))
        # self.g1 is also an identity function
        self.g1 = fn.LinearFn(np.eye(self.n1, dtype=np.float32))
        self.lnL = 0
        self.err2 = 0
        self.id0 = 0
        self.id1 = 0

    def __call__(self, inputs):
        """The main routine that is called from brica.
        The input variable  'inputs'  is a python dictionary object that brings all the current states of bundles.
        Ex.

          inputs = {"Bundle0", st0, "Bundle1", st1}

        The output variable 'results' is a python dictionary object which brings all the feedback states to the corresponding bundles.
        Ex.

          results = {"Bundle0", fbst0, "Bundle1", fbst1}

        """
        self.update(inputs)
        return self.results

    def _state_record(self):
        """Storing records of current MatcherEKF
        """
        b0 = self.b0state
        b1 = self.b1state
        fbst0 = self.results[self.b0name]
        fbst1 = self.results[self.b1name]
        # feedback from the current Matcher to the Bundle b1
        mu0 = np.array(b0.data["mu"])
        sigma0 = np.array([np.diag(b0.data["Sigma"])], dtype=np.float32)
        mu1 = np.array(b1.data["mu"])
        sigma1 = np.array([np.diag(b1.data["Sigma"])], dtype=np.float32)
        dmu1 = np.array(fbst1.data["mu"])
        dsigma1 = np.array([np.diag(fbst1.data["Sigma"])], dtype=np.float32)

        if self._first_call_of_state_record:
            self.record = {
                "mu0": mu0,
                "diagSigma0": sigma0,
                "mu1": mu1,
                "diagSigma1": sigma1,
                "dmu1": dmu1,
                "diagDSigma1": dsigma1
            }
            self._first_call_of_state_record = False
        else:
            self.record["mu0"] = np.vstack((self.record["mu0"], mu0))
            self.record["diagSigma0"] = np.concatenate((self.record["diagSigma0"], sigma0), axis=0)
            self.record["mu1"] = np.vstack((self.record["mu1"], mu1))
            self.record["diagSigma1"] = np.concatenate((self.record["diagSigma1"], sigma1), axis=0)
            self.record["dmu1"] = np.vstack((self.record["dmu1"], dmu1))
            self.record["diagDSigma1"] = np.concatenate((self.record["diagDSigma1"], dsigma1), axis=0)

    def forward(self):
        """Main method that evaluates the error and derivatives.
           z = g0(mu0) - g1(mu1)
           C0 = (d g0/ d mu0)   <-- Jacobian
           C1 = (d g1/ d mu1)   <-- Jacobian
           S = C0 Sigma0 C0' + C1 Sigma1 C1'
           err2 = z S^{-1} z'
           lnL = - (1/2) err2 - (1/2) log det( 2*pi*S )
           K0 = Sigma0 C0' S^{-1}
           K1 = Sigma1 C1' S^{-1}
           dmu0 <-- K0 z,  dSigma0 <-- K0 C0 dSigma0
           dmu1 <-- K1 z,  dSigma1 <-- K1 C1 dSigma1

        Note
           When bundle  b0  is a Observer that provides sequencial data, the user may consider the simplest setting with the identity function  g0()  and a fixed noise matrix  Sigma0. In this case, the observer omits the corresponding feedback signal  dmu0  and  dSigma0, and the total covariance  S  becomes that of a standard EKF,
               S = R + C1 Sigma1 C1
           where  R = Sigma0.
           In other words, user may provide a fixed noise matrix  Sigma0 = R  in the observer in order to set the observation noise model.
           Missing observation can be described as temporally setting of large diagonal elements of  Sigma0.
        """
        self.logger.debug("Matcher_EKF forward")
        self.lnL_t = 0
        # self.R = self.Sigma0 + self.Sigma1
        z = self.g0.value(self.mu0) - self.g1.value(self.mu1)
        C0 = self.g0.x(self.mu0)
        C1 = self.g1.x(self.mu1)
        S = np.dot(np.dot(C0.T, self.Sigma0), C0) + np.dot(np.dot(C1.T, self.Sigma1), C1)
        SI = np.linalg.inv(S)
        dum_sign, logdet = np.linalg.slogdet(S)
        self.lnL_t -= np.dot(np.dot(z, SI), z.T) / 2.0
        self.err2 += np.dot(z, z.T)
        self.lnL_t -= logdet / 2.0
        K0 = np.dot(np.dot(self.Sigma0, C0), SI)
        K1 = np.dot(np.dot(self.Sigma1, C1), SI)

        self.dmu0 = -np.dot(K0, z)  #### HERE it is fixed! ####
        self.dmu1 = np.dot(K1, z)
        self.dSigma0 = -np.dot(K0, np.dot(C0.T, self.Sigma0))
        self.dSigma1 = -np.dot(K1, np.dot(C1.T, self.Sigma1))
        self.lnL += self.lnL_t
        self.logger.debug("lnL_t = {lnLt}, lnL = {lnL}".format(lnLt=self.lnL_t, lnL=self.lnL))

    def backward(self):
        """Updates the observation models
              self.g0 and self.g1
        if they are variables.
        (to be implemented soon)
        """
        self.logger.debug("{} backward".format(self.name))

    def update(self, inputs):
        """method self.update()
         is called from self.__call__()
        """
        self.b0state, self.b1state = inputs[self.b0name], inputs[self.b1name]
        d0, d1 = self.b0state.data, self.b1state.data

        self.mu0 = d0["mu"]
        self.Sigma0 = d0["Sigma"]
        self.ts0 = d0["time_stamp"]
        self.mu1 = d1["mu"]
        self.Sigma1 = d1["Sigma"]
        self.ts1 = d1["time_stamp"]

        self.logger.debug("mu0={}".format(self.mu0))
        self.logger.debug("Sigma0={}".format(self.Sigma0))
        self.logger.debug("mu1={}".format(self.mu1))
        self.logger.debug("Sigma1={}".format(self.Sigma1))

        if self.ts0 == self.ts0_recent:
            self.logger.debug("b0.state is not updated")
        if self.ts1 == self.ts1_recent:
            self.logger.debug("b1.state is not updated")
        self.ts0_recent = self.ts0
        self.ts1_recent = self.ts1

        self.forward()
        self.backward()

        self.results[self.b0name].data["mu"] = self.dmu0.astype(np.float32)
        self.results[self.b0name].data["Sigma"] = self.dSigma0.astype(np.float32)
        self.results[self.b1name].data["mu"] = self.dmu1.astype(np.float32)
        self.results[self.b1name].data["Sigma"] = self.dSigma1.astype(np.float32)