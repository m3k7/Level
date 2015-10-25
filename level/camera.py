import cv2
import time
import numpy as np
from pycontrib.misc.informer import Informer

class Camera(object):
    
    def __init__(self, **kwargs):
        self.usePiCamera = kwargs.get('pi_camera', False)
#         if self.usePiCamera:
#             import picamera
#             import picamera.array
            
        self.resolutionDiv = int(kwargs.get('resolution_div', 2))
        self.framerate = int(kwargs.get('faramerate', 2))
        if self.usePiCamera:
            self.camera = picamera.PiCamera()
            self.camera.resolution = np.array((2592, 1944))//self.resolutionDiv
            self.camera.framerate = self.framerate
            time.sleep(2)
            self.capture = picamera.array.PiRGBArray(self.camera)
        else:
            self.fn = kwargs.get('file', 0)
            self.capture = cv2.VideoCapture(self.fn)
            
    def stream(self):
        if self.usePiCamera:
            for frame in self.camera.capture_continuous(self.capture, format="bgr", use_video_port=True):
                yield 1, frame.array
                self.capture.truncate(0)
        else:
            while 1:
#                 Informer.info('Image captured')
                yield self.capture.read()
        yield 0, None
