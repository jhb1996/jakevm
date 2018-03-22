#!/usr/bin/env python3
import numpy as np
import sys, os, imp
import cv2
import transformations
import traceback
from features import *
from scipy import signal


if __name__ == "__main__":
    #print ("myTest works")
    #HKD = HarrisKeypointDetector()
    #print(ndimage.filters.gaussian_filter(25, .5))
    #print (HKD.computeHarrisValues(np.array([[1,1,1],[0,0,0],[1,1,1]])))
    #m_image = np.array([[1,0,0],[0,0,0],[0,0,0]])
    #HKD = HarrisKeypointDetector()
    #maxes = HKD.computeLocalMaxima(m_image)
    #print (maxes)
    #
    #print(HKD.detectKeypoints(np.array([[[1,1,1],[1,1,1],[1,1,1]],[[0,0,0],[0,0,0],[0,0,0]],[[1,1,1],[1,1,1],[1,1,1]]])))
    x=40
    y=40
    angle = 2
    T1 = transformations.get_trans_mx(np.array([-x,-y,0]))
    R  = transformations.get_rot_mx(0, 0, -angle)
    S  = transformations.get_scale_mx(.2, .2, 0)
    T2 = transformations.get_trans_mx(np.array([4,-4,0]))
    four_x_four=np.dot(np.dot(np.dot(T2, S), R),T1)
            
    transMx = (four_x_four[0:2, [0, 1, 3]])
    print(transMx)