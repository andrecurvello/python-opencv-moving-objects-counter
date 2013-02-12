#!/usr/bin/python
import cv2
import numpy as np
import video_capturer
import logging
import itertools
LOG = logging.getLogger()

def color2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
def features_to_track(img, maxcorners=128, qualitylevel=.01, mindistance=4, mask=None):
    tracking_features = cv2.goodFeaturesToTrack( color2gray(img)
                                                 , maxcorners
                                                 , qualitylevel
                                                 , mindistance
                                                 , mask=mask )
    if tracking_features != None:
        tracking_features = tracking_features.reshape((-1, 2))
    return tracking_features

def optical_flow(fgrayprev, fgray, tracking_features):
    winsize = 10
    if tracking_features != None:
        forwardflow, status, track_error \
            = cv2.calcOpticalFlowPyrLK( fgrayprev
                                        , fgray
                                        , tracking_features
                                        , None
                                        , winSize=(winsize,winsize)
                                        , maxLevel=5
                                        , criteria = (cv2.TERM_CRITERIA_EPS \
                                                          | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        backflow, backstatus, backtrack_error \
            = cv2.calcOpticalFlowPyrLK( fgray
                                        , fgrayprev
                                        , forwardflow
                                        , None
                                        , winSize=(winsize,winsize)
                                        , maxLevel=5
                                        , criteria = (cv2.TERM_CRITERIA_EPS \
                                                          | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        d = abs(tracking_features-backflow).reshape(-1, 2).max(-1)
        good = d < 1
        forwardflow = forwardflow.reshape((-1, 2))
        finalflow = []
        finalfeat = []
        for feat,flow,qualitygood in zip(tracking_features, forwardflow, good):
            if qualitygood:
                finalflow.append(flow)
                finalfeat.append(feat)
        return finalflow,finalfeat
    else:
        return None,None

def clean_flow(flows, feats, minspeed=2.0):
    cleanflow = []
    cleanfeat = []
    for feat,flow in zip(flows, feats):
        x1,y1 = feat[0],feat[1]
        x2,y2 = flow[0],flow[1]
        speed = np.sqrt((x2-x1)**2+(y2-y1)**2)
        if speed > minspeed:
            cleanfeat.append(feat)
            cleanflow.append(flow)
    return cleanfeat,cleanflow

def compute_mean_speed(flows, feats):
    speedsum=0
    speednum=0
    if flows != None and flows != [] and feats != None and feats != []:
        flowsarr = np.asarray(flows)
        featsarr = np.asarray(feats)
        speeds = np.sqrt((flowsarr[:,0]-featsarr[:,0])**2+(flowsarr[:,1]-featsarr[:,1])**2)
        mean_speed = np.mean(speeds)
    else:
        mean_speed = None
    return mean_speed

def visualize_flow(flows, feats, img):
    for feat,flow in zip(flows, feats):
        x1,y1 = feat[0],feat[1]
        x2,y2 = flow[0],flow[1]
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
        cv2.circle(img,(x2,y2),2,(0,0,255))
    return img

def main( frame_resize_factor=None, threshold=20, initiate_background=False ):
    with video_capturer.video_capturer( frame_resize_factor = frame_resize_factor
                                        , initiate_background = initiate_background
                                        ) as v:
        fgrayprev = color2gray(v.get_frame())
        while( True ):
            frame = v.get_frame()
            
            background = v.get_background()
            
            dif = cv2.absdiff(background,frame)
            res2 = cv2.convertScaleAbs(dif)
            res2 = color2gray(res2)
            
            _,thresh_frame = cv2.threshold(res2, threshold, 255, cv2.THRESH_BINARY)

            fgray = color2gray(frame)
            tracking_features = features_to_track(frame, mask=thresh_frame)
            flow,feat = optical_flow(fgrayprev, fgray, tracking_features)
            if flow != None and feat != None:
                flow,feat = clean_flow(flow, feat, minspeed=2.0)
            if flow != None and feat != None:
                frame = visualize_flow(flow, feat, frame)
                mean_speed = compute_mean_speed(flow, feat)
                if mean_speed != None: LOG.info("mean_speed=%f" % mean_speed)
            fgrayprev = fgray

            
            cv2.imshow('img',frame);
            cv2.imshow('background',background)
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
