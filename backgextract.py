#!/usr/bin/python
import cv2
import numpy as np
c = cv2.VideoCapture(0)
_,f = c.read()

def color2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def resize(img,factor):
    imgsize=img.shape
    imgwidth=int(imgsize[1]*factor)
    imgheight=int(imgsize[0]*factor)
    return cv2.resize(img,(imgwidth,imgheight))

def preprocess(img):
    smooth = cv2.GaussianBlur(img,(11,11),0)
    return smooth

factor = 0.5

f = resize(f, factor)
avg = np.float32(f)
dif = np.float32(f)

numchanged = 0
numdetect = 0
while(1):
    _,f = c.read()

    f = resize(f,factor)
    f = preprocess(f)

    numchangedthreshold = 4000
    if numchanged > numchangedthreshold:
        alpha = 0.01
        if event:
            numdetect += 1
            print("numdetect=%i" % numdetect)
            event = False
    else:
        alpha = 0.5
        event = True

    cv2.accumulateWeighted(f,avg,alpha)
    res1 = cv2.convertScaleAbs(avg)
 
    dif = cv2.absdiff(res1,f)
    res2 = cv2.convertScaleAbs(dif);
    res2 = color2gray(res2)

    _,thresh = cv2.threshold(res2, 20, 255, cv2.THRESH_BINARY)

    numchanged = cv2.countNonZero(thresh) 

    cv2.imshow('img',f);
    cv2.imshow('avg',res1)
    cv2.imshow('res',thresh)

    k = cv2.waitKey(20)
 
    if k == 27:
        break
 
cv2.destroyAllWindows()
c.release()
