import numpy as np
import unittest

from matchernet.ekf import BundleEKFContinuousTime
from matchernet import fn
from matchernet import utils


class TestBundleEKFContinuousTime(unittest.TestCase):
    def setUp(self):
        self.n = 2
        self.A = np.array([[-0.1, 2], [-2, -0.1]], dtype=np.float32)
        self.b0 = BundleEKFContinuousTime("b0", 2, fn.LinearFn(self.A))
        self.b1 = BundleEKFContinuousTime("b1", 2, fn.LinearFn(self.A))

    def test_has_key(self):
        self.assertTrue(self.b0.state.data.keys() >= {"mu", "Sigma"})

    def test_init_val(self):
        actual = self.b0.state.data["mu"]
        expected = utils.zeros(self.n)
        self.assertIsNone(np.testing.assert_array_equal(expected, actual))


if __name__ == '__main__':
    unittest.main()
