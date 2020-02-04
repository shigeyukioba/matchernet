# -*- coding: utf-8 -*-
import numpy as np
import unittest

from camera import Camera


class CameraTest(unittest.TestCase):
    def test_init(self):
        eye_from = np.array([0.0, 0.0, -3.0], dtype=np.float32)
        eye_to   = np.array([0.0, 0.0, 0.0],  dtype=np.float32)
        
        camera = Camera(eye_from, eye_to)
        
        

if __name__ == '__main__':
    unittest.main()
    
