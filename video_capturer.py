import cv2
import logging
import numpy as np
LOG = logging.getLogger(__name__)


class video_capturer():
    def __init__( self ):
        self.c = cv2.VideoCapture(0)
        self.resize_factor = 0.5
        self.blur_kernel_size = 11 # (int(np.max(frame.shape)*0.04)/2)*2+1 # Allways an odd number
        self.alpha = 0.001
        self.background = np.float32(self._get_frame())
        self.avg = self.background

    def __enter__( self ):
        """ What to do when entering the with statement."""
        # Makes the background initialization a bit quicker. Uncomment the next line.
        # self.init_background() 
        return self

    def __exit__(self, type, value, traceback):
        """ What to do when exiting the with statement. """
        cv2.destroyAllWindows()
        self.c.release()

    def init_background(self):
        # Save original value
        orig_alpha = self.alpha
        self.alpha = 1.0  # Just some high alpha value.
        while self.alpha > orig_alpha:
            self.get_frame() # Just to get one with it...
            self.alpha -= self.alpha/10.0 # Reduce by 10%.
            LOG.debug("New alpha: '%f'"%(self.alpha))
        self.alpha = orig_alpha

    def resize(self, img, factor):
        imgsize=img.shape
        imgwidth=int(imgsize[1]*factor)
        imgheight=int(imgsize[0]*factor)
        LOG.debug( "Image resized to (%f,%f)"%(imgwidth, imgheight) )
        return cv2.resize(img,(imgwidth,imgheight))

    def preprocess(self, img):
        kernel_size = (self.blur_kernel_size,self.blur_kernel_size)
        sigma = 0
        smooth = cv2.GaussianBlur( img, kernel_size, sigma )
        return smooth

    def get_frame(self):
        frame = self._get_frame()
        self.accumulate_background( frame )
        return frame

    def _get_frame(self):
        _,frame = self.c.read()
        frame = self.resize(frame, self.resize_factor)
        frame = self.preprocess(frame)
        return frame

    def accumulate_background( self, frame ):
        cv2.accumulateWeighted( frame, self.avg, self.alpha )
        self.background = cv2.convertScaleAbs( self.avg )

    def get_background(self):
        return self.background
