import logging
from log import logging_conf
import numpy as np

from matchernet.matchernet import Bundle, Matcher
from matchernet import state
from matchernet import utils
from matchernet.model import model

logging_conf.set_logger_config("./log/logging.json")
logger = logging.getLogger(__name__)


class ImageGenerator(Bundle):
    def __init__(self, name, n, f, logger=logger):
        self.logger = logger.getChild(self.__class__.__name__)
        self.n = n  # Dimsnsionarity of the state variable
        self.name = name
        self.state = state.StateVisualMotor(n)
        self._initialize_control_params()
        self._initialize_state(n)
        self.f = f
        # self.bw = matchernet.bundleWeight(numSteps)
        self.record = {}
        self._first_call_of_state_record = True
        super(ImageGenerator, self).__init__(self.name, self.state)

    def __call__(self, inputs):
        """The main routine that is called from brica.
        """
        for key in inputs:  # key is one of the matcher names
            if inputs[key] is not None:
                self.logger.debug("accepting feedback from {}".format(key))
                self.accept_feedback(inputs[key])
        self.step_dynamics(self.dt)
        self._countup()
        self.state.data["angular_velocity"] = utils.regularize_cov_matrix(self.state.data["angular_velocity"])
        self._state_record()
        self.logger.debug("angle={}".format(self.state.data["angle"]))

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
        angle = np.array(self.state.data["angle"], dtype=np.float32)
        angular_velocity = np.array([np.diag(self.state.data["angular_velocity"])], dtype=np.float32)
        ts = np.array([self.state.data["time_stamp"]], dtype=np.float32)

        if self._first_call_of_state_record:
            self.record = {"angle": angle, "angular_velocity": angular_velocity, "time_stamp": ts}
            self._first_call_of_state_record = False
        else:
            self.record["angle"] = np.vstack((self.record["angle"], angle))
            self.record["angular_velocity"] = np.concatenate((self.record["angular_velocity"], angular_velocity),
                                                             axis=0)
            self.record["time_stamp"] = np.concatenate((self.record["time_stamp"], ts), axis=0)

    def _initialize_state(self, n):
        self.state.data["id"] = self.id
        self.state.data["angle"] = 0
        self.state.data["angular_velocity"] = 0

    def accept_feedback(self, fbst):
        d_angle = fbst.data["angle"]
        d_angular_velocity = fbst.data["angular_velocity"]
        angle = self.state.data["angle"]
        angular_velocity = self.state.data["angular_velocity"]

        self.logger.debug("d_angle={}".format(d_angle))
        self.logger.debug("d_angular_velocity={}".format(d_angular_velocity))
        weight = 1.0

        self.state.data["angle"] = (mu + weight * d_angle).astype(np.float32)
        self.state.data["angular_velocity"] = (angular_velocity + weight * d_angular_velocity).astype(np.float32)

    def step_dynamics(self, dt):
        angle = self.state.data["angle"]
        angular_velocity = self.state.data["angular_velocity"]
        Q = self.state.data["Q"]
        A = self.f.dx(angle)
        F = utils.calc_matrix_F(A, dt)
        angle = np.dot(F, angle)
        angular_velocity = dt * Q + np.dot(np.dot(F.T, angular_velocity), F)
        self.state.data["angle"] = angle
        self.state.data["angular_velocity"] = angular_velocity
        # ["time_stamp"] is updated in the method self._countup()


class MatcherEKFImage(Matcher):
    def __init__(self, name, b0, logger=logger):
        self.logger = logger.getChild(self.__class__.__name__)
        self.name = name
        super(MatcherEKFImage, self).__init__(name, b0)
        self.b0name = b0.name
        self.n0 = b0.state.n  # dim. of B0
        self._first_call_of_state_record = True
        self._initialize_model()
        self.ts_recent = -1  # the most recent value of the time_stamp of b0
        self.update_component()

    def _initialize_model(self):
        self.n = self.n0  # dim. of g0(x0), g1(x1)
        # self.g0 is an identity function as a default observation model
        self.g0 = model

    def __call__(self, inputs):
        self.update(inputs)
        return self.results

    def _state_record(self):
        """Storing records of current MatcherEKFImage
        """
        b0 = self.b0state
        fbst0 = self.results[self.b0name]
        # feedback from the current Matcher to the Bundle b1
        angle = np.array(b0.data["angle"])
        angular_velocity = np.array([np.diag(b0.data["angular_velocity"])], dtype=np.float32)

        if self._first_call_of_state_record:
            self.record = {
                "angle": angle,
                "angular_velocity": angular_velocity
            }
            self._first_call_of_state_record = False
        else:
            self.record["angle"] = np.vstack((self.record["angle"], angle))
            self.record["angular_velocity"] = np.concatenate((self.record["angular_velocity"], angular_velocity),
                                                             axis=0)

    def forward(self):
        self.logger.debug("Matcher_EKF forward")

        self.d_angle = model['angle']
        self.d_angular_velocity = model['angular_velocity']
        self.logger.debug("lnL_t = {lnLt}, lnL = {lnL}".format(lnLt=self.lnL_t, lnL=self.lnL))

    def backward(self):
        self.logger.debug("{} backward".format(self.name))

    def update(self, inputs):
        """method self.update()
         is called from self.__call__()
        """
        self.b0state = inputs[self.b0name]
        d0 = self.b0state.data

        self.angle = d0["angle"]
        self.angular_velocity = d0["angular_velocity"]
        self.ts = d0["time_stamp"]

        self.logger.debug("angle={}".format(self.angle))
        self.logger.debug("angular_velocity={}".format(self.angular_velocity))

        if self.ts == self.ts_recent:
            self.logger.debug("b0.state is not updated")
        self.ts_recent = self.ts

        self.forward()
        self.backward()

        self.results[self.b0name].data["angle"] = self.d_angle.astype(np.float32)
        self.results[self.b0name].data["angular_velocity"] = self.d_angular_velocity.astype(np.float32)
