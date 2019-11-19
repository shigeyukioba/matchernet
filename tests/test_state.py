import sys
import os
sys.path.append(os.getcwd())

import unittest

import numpy as np
from matchernet_py_001 import utils
from matchernet_py_001.state import StatePlain, StateMuSigma, StateMuSigmaDiag


class TestStatePlain(unittest.TestCase):

    def setUp(self):
        self.n = 4
        self.s = StatePlain(self.n)

    def test_has_key(self):
        self.assertTrue(self.s.data.keys() >= {"mu"})

    def test_init_val(self):
        actual = self.s.data["mu"]
        expected = utils.zeros((1, self.n))
        self.assertIsNone(np.testing.assert_array_equal(expected, actual))


class TestStateMuSigma(unittest.TestCase):

    def setUp(self):
        self.n = 4
        self.s = StateMuSigma(self.n)

    def test_has_key(self):
        self.assertTrue(self.s.data.keys() >= {"id",
                                               "time_stamp",
                                               "mu",
                                               "Sigma"})

    def test_init_val(self):
        actual_mu = self.s.data["mu"]
        expected_mu = utils.zeros((1, self.n))
        self.assertIsNone(np.testing.assert_array_equal(expected_mu, actual_mu))

        actual_sigma = self.s.data["Sigma"]
        expected_sigma = np.eye(self.n, dtype=np.float32)
        self.assertIsNone(np.testing.assert_array_equal(expected_sigma, actual_sigma))


class TestStateMuSigmaDiag(unittest.TestCase):

    def setUp(self):
        self.n = 4
        self.s = StateMuSigmaDiag(self.n)

    def test_has_key(self):
        self.assertTrue(self.s.data.keys() >= {"id",
                                               "time_stamp",
                                               "mu",
                                               "sigma"})

    def test_init_val(self):
        actual_mu = self.s.data["mu"]
        expected_mu = utils.zeros((1, self.n))
        self.assertIsNone(np.testing.assert_array_equal(expected_mu, actual_mu))

        actual_sigma = self.s.data["sigma"]
        expected_sigma = np.diag(np.eye(self.n, dtype=np.float32))
        self.assertIsNone(np.testing.assert_array_equal(expected_sigma, actual_sigma))

if __name__ == '__main__':
    unittest.main()
