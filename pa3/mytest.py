#!/usr/bin/env python3
import numpy as np
import sys, os, imp
import cv2
import transformations
import traceback
from features import *
from scipy import signal


if __name__ == "__main__":
    print ("myTest works")
    HKD = HarrisKeypointDetector()
    #print(ndimage.filters.gaussian_filter(25, .5))
    #print (HKD.computeHarrisValues(np.array([[1,1,1],[0,0,0],[1,1,1]])))
    #m_image = np.array([[1,0,0],[0,0,0],[0,0,0]])
    #HKD = HarrisKeypointDetector()
    #maxes = HKD.computeLocalMaxima(m_image)
    #print (maxes)
    print(HKD.detectKeypoints(np.array([[[1,1,1],[1,1,1],[1,1,1]],[[0,0,0],[0,0,0],[0,0,0]],[[1,1,1],[1,1,1],[1,1,1]]])))