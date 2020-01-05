import unittest
import numpy as np

from matchernet_py_001.observer import Observer, ObserverMultiple, missing_handler001


class TestMissingHandler001(unittest.TestCase):
    def test_complete(self):
        test_mu_patterns = [
            np.array([1, 2, 3]),
            np.array([1, 2, np.inf])
        ]
        for mu in test_mu_patterns:
            sigma = np.eye(len(mu))
            with self.subTest():
                actual_mu, actual_sigma = missing_handler001(mu, sigma)
                self.assertIsNone(np.testing.assert_array_equal(mu, actual_mu))
                self.assertIsNone(np.testing.assert_array_equal(sigma, actual_sigma))

    def test_missing(self):
        test_mu_patterns = [
            np.array([1, 2, np.nan]),
            np.array([np.nan, np.nan, np.nan])
        ]
        expected_mu = [0, 0, 0]
        for mu in test_mu_patterns:
            sigma = np.eye(len(mu))
            with self.subTest():
                actual_mu, actual_sigma = missing_handler001(mu, sigma)
                expected_sigma = 1000 * sigma
                self.assertIsNone(np.testing.assert_array_equal(expected_mu, actual_mu))
                self.assertIsNone(np.testing.assert_array_equal(expected_sigma, actual_sigma))


class TestObserver(unittest.TestCase):
    def setUp(self):
        self.dim = 6
        self.buffersize = 10
        self.repeat_num = 3
        self.x = np.zeros((self.buffersize, self.dim), dtype=np.float32)
        for i in range(self.buffersize):
            self.x[i][0] = i
        self.b = Observer("b0", self.x)

    def test_observer_count_up_with_brica(self):
        expected = np.zeros(self.dim, dtype=np.float32)
        for i in range(self.repeat_num):
            for j in range(self.buffersize):
                self.b.count_up()
                actual = self.b.get_state()
                expected[0] = j
                self.assertIsNone(np.testing.assert_array_equal(expected, actual))

    def test_observer_count_up(self):
        expected = np.zeros(self.dim, dtype=np.float32)
        for i in range(self.repeat_num):
            for j in range(self.buffersize):
                self.b(self.b.state)
                actual = self.b.results["state"].data["mu"]
                expected[0] = j
                self.assertIsNone(np.testing.assert_array_equal(expected, actual))

    def test_observer_multiple(self):
        mul = 3
        b_m = ObserverMultiple("b_m", self.x, mul)
        for i in range(self.repeat_num):
            for j in range(self.buffersize):
                b_m.count_up()
                actual = b_m.get_state()
                for k in range(mul):
                    _ = np.zeros(self.dim, dtype=np.float32)
                    _[0] = (j + k) % self.buffersize
                    expected = _ if k == 0 else np.concatenate([expected, _])
                self.assertIsNone(np.testing.assert_array_equal(expected, actual))


if __name__ == '__main__':
    unittest.main()
