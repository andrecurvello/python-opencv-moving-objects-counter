#!/usr/bin/python
import cv2
import numpy as np
import video_capturer
import logging
LOG = logging.getLogger()


def color2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def main( initiate_background ):
    with video_capturer.video_capturer( initiate_background = initiate_background ) as v:
        while( True ):
            f = v.get_frame()
            background = v.get_background()
            
            dif = cv2.absdiff(background,f)
            res2 = cv2.convertScaleAbs(dif)
            res2 = color2gray(res2)
            
            _,thresh = cv2.threshold(res2, 20, 255, cv2.THRESH_BINARY)
            
#            cv2.imshow('img',f);
#            cv2.imshow('background',background)
            cv2.imshow('res',thresh)

            k = cv2.waitKey(20)
            
            if k == 27:
                break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--debug", action="store_true", help="View some debugging information.")
    group.add_argument("--initiate-background", action="store_true", help="Start by initiating the background.")
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)        
    main( args.initiate_background)
