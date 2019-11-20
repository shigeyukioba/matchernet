# -*- coding: utf-8 -*-
import numpy as np
import cv2


class MovieWriter(object):
    """ Movie output utility class. """
    def __init__(self, file_name, frame_size, fps):
        """
        frame_size is (w, h)
        """
        self._frame_size = frame_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.vout = cv2.VideoWriter()
        success = self.vout.open(file_name, fourcc, fps, frame_size, True)
        if not success:
            print("Create movie failed: {0}".format(file_name))

    def add_frame(self, frame):
        """
        frame shape is (h, w, 3), dtype is np.uint8
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.vout.write(frame)

    def close(self):
        self.vout.release()
        self.vout = None
