import logging
from log import logging_conf
import numpy as np

from matchernet.observer import Observer, ObserverMultiple
from matchernet import state

logging_conf.set_logger_config("./log/logging.json")
logger = logging.getLogger(__name__)


class ImageProvider(Observer):
    def __init__(self, name, buff, logger=logger):
        self.logger = logger.getChild(self.__class__.__name__)
        self.name = name
        self.buffer = buff
        self.counter = -1
        self.length = buff.shape[0]
        self.dim = buff.shape[1]
        self.state = state.StateVisualMotor(self.dim)
        self.set_results()
        super(Observer, self).__init__(self.name, self.state)

    def __call__(self, inputs):
        self.count_up()
        self.set_results()
        return self.results

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

    def set_results(self):
        angle, angular_velocity = self.get_state()
        self.state.data["angle"] = angle
        self.state.data["angular_velocity"] = angular_velocity
        self.state.data["time_stamp"] = self.counter
        self.results = {"state": self.state}


class MultipleImageProvider(ObserverMultiple):
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
        self.state.data["angle"] = z
        return z
