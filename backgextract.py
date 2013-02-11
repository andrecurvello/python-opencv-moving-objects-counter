#!/usr/bin/python
import cv2
import numpy as np
import video_capturer
import logging
LOG = logging.getLogger()

def color2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def main( frame_resize_factor=None, threshold=20, initiate_background=False ):
    with video_capturer.video_capturer( frame_resize_factor = frame_resize_factor
                                        , initiate_background = initiate_background
                                        ) as v:
        while( True ):
            frame = v.get_frame()
            background = v.get_background()
            
            dif = cv2.absdiff(background,frame)
            res2 = cv2.convertScaleAbs(dif)
            res2 = color2gray(res2)
            
            _,thresh_frame = cv2.threshold(res2, threshold, 255, cv2.THRESH_BINARY)
            
#            cv2.imshow('img',frame);
#            cv2.imshow('background',background)
            cv2.imshow('res',thresh_frame)

            k = cv2.waitKey(20)
            
            if k == 27:
                break

if __name__ == "__main__":
    import argparse

    def frame_size_factor( factor ):
        LOG.debug("Frame resize factor: '%s'."%(factor))
        factor = float(factor)
        if factor <= 0 or factor > 1:
            raise argparse.ArgumentError("frame size factor, '%f', must be between 0 and 1."%(factor))
        return factor

    def threshold( threshold ):
        LOG.debug("Threshold: '%s'."%(threshold))
        threshold = float(threshold)
        if threshold <= 0 or threshold > 255:
            raise argparse.ArgumentError("Threshold, '%f', must be between 0.0 and 255.0."%(threshold))
        return threshold

    parser = argparse.ArgumentParser()
    parser.add_argument( "-d", "--debug"
                         , action="store_true"
                         , help="View some debugging information." )
    parser.add_argument( "--initiate-background"
                         , action="store_true"
                         , help="Start by initiating the background. This creates a usable background much faster." )
    parser.add_argument( "--frame-resize-factor"
                         , type=frame_size_factor
                         , help="Resize the frame by this factor. Somewhere between 0 and 1." )
    parser.add_argument( "-t", "--threshold"
                         , type=threshold
                         , help="Threshold for how much a pixel may differ from the background before it is concidered a change. 0 to 255."
                         , default=20 )

    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        LOG.debug("Logging set to debug.")
    else:
        logging.basicConfig(level=logging.INFO)

    main( frame_resize_factor = args.frame_resize_factor
          , initiate_background = args.initiate_background
          , threshold = args.threshold
          )
