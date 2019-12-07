import unittest

import numpy as np
from matchernet_py_001.fn import LinearFn, LinearFnXU


class TestLinearFn(unittest.TestCase):

    def setUp(self):
        self.x = np.array([10, 20, 30], dtype=np.float32)
        self.test_A_patterns = [
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
            np.array([[1, 2, np.inf]], dtype=np.float32)
        ]

    def test_linear_dot(self):
        for A in self.test_A_patterns:
            f = LinearFn(A)
            expected_dot = np.dot(A, self.x)
            with self.subTest():
                actual_dot = f.value(self.x)
                self.assertIsNone(np.testing.assert_array_equal(expected_dot, actual_dot))

    def test_linear_dx(self):
        for A in self.test_A_patterns:
            f = LinearFn(A)
            expected_dx = A
            with self.subTest():
                actual_dx = f.dx(self.x)
                self.assertIsNone(np.testing.assert_array_equal(expected_dx, actual_dx))


class TestLinearFnXU(unittest.TestCase):

    def setUp(self):
        self.x = np.array([10, 20, 30], dtype=np.float32)
        self.u = np.array([5, 15, 25], dtype=np.float32)
        self.test_A_patterns = [
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
            np.array([[1, 2, np.inf]], dtype=np.float32)
        ]
        self.test_B_patterns = [
            np.array([[6, 5, 4], [3, 2, 1]], dtype=np.float32),
            np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float32),
            np.array([[np.inf, 2, 3]], dtype=np.float32)
        ]

    def test_linear_XU_dot(self):
        for A, B in zip(self.test_A_patterns, self.test_B_patterns):
            f = LinearFnXU(A, B)
            expected_dot = np.dot(A, self.x) + np.dot(B, self.u)
            with self.subTest():
                actual_dot = f.value(self.x, self.u)
                self.assertIsNone(np.testing.assert_array_equal(expected_dot, actual_dot))

    def test_linear_XU_dx(self):
        for A, B in zip(self.test_A_patterns, self.test_B_patterns):
            f = LinearFnXU(A, B)
            expected_dx = A
            with self.subTest():
                actual_dx = f.dx(self.x, self.u)
                self.assertIsNone(np.testing.assert_array_equal(expected_dx, actual_dx))

    def test_linear_XU_du(self):
        for A, B in zip(self.test_A_patterns, self.test_B_patterns):
            f = LinearFnXU(A, B)
            expected_du = B
            with self.subTest():
                actual_du = f.du(self.x, self.u)
                self.assertIsNone(np.testing.assert_array_equal(expected_du, actual_du))

if __name__ == '__main__':
    unittest.main()
