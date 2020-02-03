import unittest
import numpy as np

from matchernet import utils
from matchernet.state import StatePlain, StateMuSigma, StateMuSigmaDiag


class TestStatePlain(unittest.TestCase):
    def setUp(self):
        self.n = 4
        mu = np.zeros(self.n, dtype=np.float32)
        self.s = StatePlain(mu)

    def test_has_key(self):
        self.assertTrue(self.s.data.keys() >= {"mu"})

    def test_init_val(self):
        actual = self.s.data["mu"]
        expected = utils.zeros(self.n)
        self.assertIsNone(np.testing.assert_array_equal(expected, actual))


class TestStateMuSigma(unittest.TestCase):
    def setUp(self):
        self.n = 4
        mu = np.zeros(self.n, dtype=np.float32)
        Sigma = np.eye(self.n, dtype=np.float32)
        self.s = StateMuSigma(mu, Sigma)

    def test_has_key(self):
        self.assertTrue(self.s.data.keys() >= {"mu", "Sigma"})

    def test_init_val(self):
        actual_mu = self.s.data["mu"]
        expected_mu = utils.zeros(self.n)
        self.assertIsNone(np.testing.assert_array_equal(expected_mu, actual_mu))

        actual_sigma = self.s.data["Sigma"]
        expected_sigma = np.eye(self.n, dtype=np.float32)
        self.assertIsNone(np.testing.assert_array_equal(expected_sigma, actual_sigma))


class TestStateMuSigmaDiag(unittest.TestCase):
    def setUp(self):
        self.n = 4
        mu = np.zeros(self.n, dtype=np.float32)
        sigma = np.diag(np.eye(self.n, dtype=np.float32))
        self.s = StateMuSigmaDiag(mu, sigma)

    def test_has_key(self):
        self.assertTrue(self.s.data.keys() >= {"mu", "sigma"})

    def test_init_val(self):
        actual_mu = self.s.data["mu"]
        expected_mu = utils.zeros(self.n)
        self.assertIsNone(np.testing.assert_array_equal(expected_mu, actual_mu))

        actual_sigma = self.s.data["sigma"]
        expected_sigma = np.diag(np.eye(self.n, dtype=np.float32))
        self.assertIsNone(np.testing.assert_array_equal(expected_sigma, actual_sigma))


if __name__ == '__main__':
    unittest.main()
