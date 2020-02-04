import logging
from log import logging_conf
import numpy as np

from matchernet import matchernet

logging_conf.set_logger_config("./log/logging.json")
logger = logging.getLogger(__name__)


class Observer(matchernet.Bundle):
    """Observer works as a Bundle that provides vector data at each time step of the MatcherNet simulation.

    Usage:
    Construct an observer by

    >> o1 = Observer("Observer1", buffer)

    where buffer is an numpy.ndarray of shape (length, dim).
    Then, call it for each step by

    >> result = o1( dummy_input )

    and you get the vector data of the current time-stamp

     results.state["mu"]

    with corresponding observation error covariance matrix

     results.state["Sigma"].

    Note:
    When the vector data in buffer included  NaN  entries, they are regarded as missing entries and the Observer outpus a zero vector  mu  with covariance matrix  cov  of large eigen values. (See the function  missing_handler001()  for a default setting to construct the corresponding output. )
    """

    def __init__(self, name, buff, obs_noise_covariance=None, logger=logger):
        self.logger = logger.getChild(self.__class__.__name__)
        self.name = name
        self.buffer = buff
        self.counter = -1
        self.length = buff.shape[0]
        self.dim = buff.shape[1]

        if obs_noise_covariance is None:
            self.obs_noise_covariance = 1000 * np.eye(self.dim, dtype=np.float32)
        else:
            self.obs_noise_covariance = obs_noise_covariance
        self.missing_handler = missing_handler
        # default setting of missing value handler function
        # TODO:
        #self.set_results()
        # set the first value with large obs_noise_covariance
        # for an initial value
        super(Observer, self).__init__(self.name)

    def __call__(self, inputs):
        """ The main routine that is called from brica.
        """
        self.count_up()
        return self.get_results()

    def count_up(self):
        self.counter = (self.counter + 1) % self.length

    def set_buffer(self, buff):
        self.buffer = buff
        self.counter = -1
        self.length = buff.shape[0]
        self.dim = buff.shape[1]

    def get_buffer(self):
        return self.buffer

    def get_state(self):
        b = self.get_buffer()
        z = b[self.counter].copy()
        return z

    def get_results(self):
        q = self.get_state()
        mu, Sigma = self.missing_handler(np.array(q, dtype=np.float32),
                                         self.obs_noise_covariance, self.logger)

        results = {}
        results["mu"] = mu
        results["Sigma"] = Sigma
        #results["time_stamp"] = self.counter
        results["time_id"] = self.counter
        return {
            "state" : results
        }
        # === Note: We may regard  "time_stamp"  as a real time rather than a counter in a future version.


def missing_handler(mu, Sigma, logger=logger):
    """A missing value handler function.
    It receives a vector data  mu  with a default covariance matrix  Sigma, find NaN in the vector  mu, and outputs a modified set of a vector  mu  and a covariance  cov.
    """
    if np.any(np.isnan(mu)):
        logger.debug("Missing!")
        cov = Sigma * 1000
        mu = np.zeros(mu.shape)
    else:
        cov = Sigma
    return mu, cov


class ObserverMultiple(Observer):
    """A bundle that provides sequencial data
    """

    def __init__(self, name, buff, mul):
        self.name = name
        self.mul = mul
        super().__init__(self.name, buff)

    def get_state(self):
        b = self.get_buffer()
        z = b[self.counter].copy()
        for i in range(1, self.mul):
            j = (self.counter + i) % self.length
            z = np.concatenate((z, b[j].copy()))
        return z
