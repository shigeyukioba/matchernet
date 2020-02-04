# -*- coding: utf-8 -*-
import numpy as np
import os
import shutil
import cv2
import subprocess


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


class AnimGIFWriter(object):
    """ Anim GIF output utility class. """
    def __init__(self, file_name, fps, tmp_dir_name="tmp_gif"):
        """
        frame_size is (w, h)
        """
        self.file_name = file_name
        self.fps = fps
        self.tmp_dir_name = tmp_dir_name
        self.image_index = 0

        if os.path.exists(self.tmp_dir_name):
            shutil.rmtree(self.tmp_dir_name)
        os.mkdir(self.tmp_dir_name)

    def add_frame(self, frame):
        """
        frame shape is (h, w, 3), dtype is np.uint8
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        file_path = os.path.join(self.tmp_dir_name, "frame{0:03d}.png".format(self.image_index))
        cv2.imwrite(file_path, frame)
        self.image_index += 1
        
    def close(self):
        delay = int(round(100 / self.fps))
        command = "convert -delay {} {}/*.png {}".format(delay,
                                                         self.tmp_dir_name,
                                                         self.file_name)
        subprocess.call(command.split())
        shutil.rmtree(self.tmp_dir_name)
