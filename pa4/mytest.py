#!/usr/bin/env python3
import numpy as np
from student import *

def main():
    lights = np.array([[1,2,3],[9,5,1],[8,2,3],[4,6,7]])
    
    images = np.array([[1,2,3,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    
    G = compute_photometric_stereo_impl(lights, images)
    print (g)
    
    

if __name__ == "__main__":
    main()
