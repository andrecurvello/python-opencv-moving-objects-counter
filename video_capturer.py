import cv2
import logging
LOG = logging.getLogger(__name__)


class video_capturer():
    def __enter__(self):
        self.c = cv2.VideoCapture(0)
        self.resize_factor = 0.5
        self.blur_kernel_size = 11 # (int(np.max(frame.shape)*0.04)/2)*2+1 # Allways an odd number
        return self

    def __exit__(self, type, value, traceback):
        cv2.destroyAllWindows()
        self.c.release()

    def get_frame(self):
        _,frame = self.c.read()
        frame = self.resize(frame, self.resize_factor)
        frame = self.preprocess(frame)
        return frame

    def resize(self, img, factor):
        imgsize=img.shape
        LOG.debug( imgsize )
        imgwidth=int(imgsize[1]*factor)
        imgheight=int(imgsize[0]*factor)
        return cv2.resize(img,(imgwidth,imgheight))

    def preprocess(self, img):
        LOG.debug( self.blur_kernel_size )
        kernel_size = (self.blur_kernel_size,self.blur_kernel_size)
        sigma = 0
        smooth = cv2.GaussianBlur( img, kernel_size, sigma )
        return smooth
