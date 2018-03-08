#!/usr/bin/env python3
from segment import *

import math
import numpy as np
import scipy
import scipy.sparse
from scipy.sparse import spdiags
from scipy.signal import convolve2d
import cv2
from colors import COLORS
#######delete this#######
from scipy import ndimage
#########################

if __name__ == "__main__":
    # cvImage = np.array([[10,11,12],[13,14,15],[16,17,18]])
    minIn = 10
    maxIn = 20
    minOut = -500
    maxOut = -200
    #print (normalizeImage(cvImage, minIn, maxIn, minOut, maxOut))
    gradientImage = np.array([[-0,-5,-20],[5,10,20],[-7,0,7]])
    #print(getDisplayGradient(gradientImage))
    #imageForGradient = np.array([[0,5,10],[5,50,70],[80,90,100]])
    # imageForGradient = np.array([[9,9,9],[9,9,9],[9,9,9]])
    imageForGradient = np.array([[0,0,0],[0,0,0],[0,0,0]])
    

    #print(takeYGradient(imageForGradient))
    #processed = ndimage.sobel(imageForGradient, axis=0, mode = 'constant', cval=0.0)
    #print(processed)
    
    #print(takeGradientMag(imageForGradient ))
    #print (chooseRandomCenters(cvImage, 1))
    
    #print(cvImage[ : , :2])
    
    # cvImage = np.array([[0,0,0],[0,1,0],[0,0,0]])
    # print(takeGradientMag(cvImage))
    
    #pixelList = np.array([[200,0,0,100,100,100],[200,0,0,100,100,100],[0,0,100,100,200,100],[100.9,0,0,100,100,500],[0,0,0,100,100,100],[0,0.1,0.2,1,1,1],[0,0.1,0.2,1,1,1],[0,0.1,0.2,1,1,1],[0,0.1,0.2,1,1,1]])
    #print(kMeansSolver(pixelList, 3, centers=None, eps=0.001, maxIterations=100))
    
    # cvImage = np.array([[3,8,0],[0,1,0],[0,0,0]])
    # W = getColorWeights(cvImage, 1, sigmaF=1, sigmaX=1)
    #print (W)
    
    # imageForGradient = np.array([  [[1,0,0],[0,1,0],[0,0,9]],[[1,0,0],[0,1,0],[0,0,9]],[[1,0,0],[0,1,0],[0,0,9]]   ])
# 
# 
    #print(takeXGradient(imageForGradient))
    #processed = ndimage.sobel(imageForGradient, axis=1, mode = 'constant', cval=0.0)
    #print(processed)
    
    #print (takeGradientMag(imageForGradient))
    
    cvImage = np.array([[[100,200,200],[0,200,200],[0,0,0]],[[200,290,200],[200,200,200],[200,200,200]],[[100,90,100],[100,100,100],[100,100,100]]])
    cvImage2 = np.array([[[100,90,100],[100,100,100],[100,100,100]],[[100,90,100],[100,100,100],[100,100,100]]])

    
    W = getColorWeights(cvImage2, 10, sigmaF=1, sigmaX=1)
    print (W)
    #print (W)
    #d = getTotalNodeWeights(W)
    #print(d)
    #y = approxNormalizedBisect(W, d)
    #print(y)
    #segments = reconstructNCutSegments(cvImage, y, 0)

