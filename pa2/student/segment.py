# segment.py - Segments an input image.
# Cornell University CS 4670/5670: Intro Computer Vision
import math
import numpy as np
import scipy
import scipy.sparse
from scipy.sparse import spdiags
from scipy.signal import convolve2d
import cv2
from colors import COLORS



#########################################################
###    Part A: Image Processing Functions
######################################################### 

# TODO:PA2 Fill in this function
def normalizeImage(cvImage, minIn, maxIn, minOut, maxOut):
    '''
    Take image and map its values linearly from [min_in, max_in]
    to [min_out, max_out]. Assume the image does not contain 
    values outside of the [min_in, max_in] range.

    Parameters:
    cvImage - a (m x n) or (m x n x 3) image.
    minIn - the minimum of the input range
    maxIn - the maximum of the input range
    minOut - the minimum of the output range
    maxOut - the maximum of the output range

    Return:
    renormalized - the linearly rescaled version of cvImage.
    '''
    shape = np.shape(cvImage)
    if len(shape)==2:     
        m,n = np.shape(cvImage)
        new = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                new[i][j]= (((((float(cvImage[i][j])-minIn)/float(maxIn-minIn)))*float(maxOut-minOut))+minOut)
        return new
    else:#color image
        m,n,_ = np.shape(cvImage)
        new = np.zeros((m,n,3))
        for i in range(m):
            for j in range(n):
                new[i][j][0]= (((((float(cvImage[i][j][0])-minIn)/float(maxIn-minIn)))*float(maxOut-minOut))+minOut)
                new[i][j][1]= (((((float(cvImage[i][j][1])-minIn)/float(maxIn-minIn)))*float(maxOut-minOut))+minOut)
                new[i][j][2]= (((((float(cvImage[i][j][2])-minIn)/float(maxIn-minIn)))*float(maxOut-minOut))+minOut)
        return new
# TODO:PA2 Fill in this function
def getDisplayGradient(gradientImage):
    """
    Use the normalizeImage function to map a 2d gradient with one
    or more channels such that where the gradient is zero, the image
    has 50% percent brightness. Brightness should be a linear function 
    of the input pixel value. You should not clamp, and 
    the output pixels should not fall outside of the range of the uint8 
    datatype.
    
    Parameters:
        gradientImage - a per-pixel or per-pixel-channel gradient array
        either (m x n) or (m x n x 3). May have any 
        numerical datatype.

    Return:
        displayGrad - a rescaled image array with a uint8 datatype.
    """
    maximum = np.amax(gradientImage)
    minimum = np.amin(gradientImage)
    extreme = max(abs(maximum), abs(minimum))
    
    maxIn = extreme 
    minIn = extreme*-1 #is the max and min just the positive and negative of the extreme values?
    normalized = normalizeImage(gradientImage, minIn, maxIn, 0, 255)
    return normalized.astype(np.uint8)
    
# TODO:PA2 Fill in this function
def takeXGradient(cvImage):
    '''
    Compute the x-derivative of the input image with an appropriate
    Sobel implementation. Should return an array made of floating 
    point numbers.
    
    Parameters:
        cvImage - an (m x n) or (m x n x 3) image
        
    Return:
        xGrad - the derivative of cvImage at each position w.r.t. the x axis.
    
    '''
    cvImage32 = cvImage.astype(np.float32)
    filter1 = np.array([[1], [2], [1]], dtype = np.float32)
    filter2 = np.array([[1, 0, -1]], dtype = np.float32)
    shape = np.shape(cvImage32)
    if len(shape)==2:     
        m,n = np.shape(cvImage32)
        conv1 = convolve2d(cvImage32, filter1, mode='same', boundary='fill', fillvalue=0)
        conv2 = convolve2d(conv1, filter2, mode='same', boundary='fill', fillvalue=0)
        return conv2
    else:#color image
        #convolve each channel seperately
        b,g,r = cv2.split(cvImage32)
        conv1_b = convolve2d(b,       filter1, mode='same', boundary='fill', fillvalue=0)
        conv2_b = convolve2d(conv1_b, filter2, mode='same', boundary='fill', fillvalue=0)
        conv1_g = convolve2d(g,       filter1, mode='same', boundary='fill', fillvalue=0)
        conv2_g = convolve2d(conv1_g, filter2, mode='same', boundary='fill', fillvalue=0)
        conv1_r = convolve2d(r,       filter1, mode='same', boundary='fill', fillvalue=0)
        conv2_r = convolve2d(conv1_r, filter2, mode='same', boundary='fill', fillvalue=0)
        img = cv2.merge((conv2_b,conv2_g,conv2_r))
        return img

    # TODO:PA2 Fill in this function
def takeYGradient(cvImage):
    # '''
    # Compute the y-derivative of the input image with an appropriate
    # Sobel implementation. Should return an array made of floating 
    # point numbers.
    # 
    # Parameters:
    #     cvImage - an (m x n) or (m x n x 3) image
    # 
    # Return:
    # yGrad - the derivative of cvImage at each position w.r.t. the y axis.
    # '''
    cvImage32 = cvImage.astype(np.float32)
    filter1 = np.array([[1], [2], [1]], dtype = np.float32)
    filter2 = np.array([[1, 0, -1]], dtype = np.float32)
    shape = np.shape(cvImage32)
    if len(shape)==2:     
        m,n = np.shape(cvImage32)
        conv1 = convolve2d(cvImage32, filter1, mode='same', boundary='fill', fillvalue=0)
        conv2 = convolve2d(conv1, filter2, mode='same', boundary='fill', fillvalue=0)
        return conv2
    else:#color image
        #convolve each channel seperately
        b,g,r = cvImage32[:,:,0],cvImage32[:,:,1],cvImage32[:,:,2] #cv2.split(cvImage32)
        conv1_b = convolve2d(b,       filter1, mode='same', boundary='fill', fillvalue=0)
        conv2_b = convolve2d(conv1_b, filter2, mode='same', boundary='fill', fillvalue=0)
        conv1_g = convolve2d(g,       filter1, mode='same', boundary='fill', fillvalue=0)
        conv2_g = convolve2d(conv1_g, filter2, mode='same', boundary='fill', fillvalue=0)
        conv1_r = convolve2d(r,       filter1, mode='same', boundary='fill', fillvalue=0)
        conv2_r = convolve2d(conv1_r, filter2, mode='same', boundary='fill', fillvalue=0)
        img = np.zeros(shape,dtype=np.float32)#cv2.merge((conv2_b,conv2_g,conv2_r))
        img [:,:,0] = conv2_b
        img [:,:,1] = conv2_g
        img [:,:,2] = conv2_r
        return img
    # # TODO:PA2 Fill in this function
def takeGradientMag(cvImage):
    '''
    Compute the gradient magnitude of the input image for each 
    pixel in the image. 
    
    Parameters:
        cvImage - an (m x n) or (m x n x 3) image
        
    Return:
        gradMag - the magnitude of the 2D gradient of cvImage. 
    if multiple channels, handle each channel seperately.
    '''
    #take the gradient with respect to x and y and then find the magnitude of the gradient (by taking the dot product of it with itself)
    shape = np.shape(cvImage)
    if len(shape)==2:
        grad_x = (takeXGradient(cvImage))
        grad_x_sq = np.power(grad_x,2)
        grad_y = (takeYGradient(cvImage))
        grad_y_sq = np.power(grad_y,2)
        return np.power(np.add(grad_x_sq, grad_y_sq), .5)

    else:#color image
        #convolve each channel seperately
        grad_x = (takeXGradient(cvImage))
        grad_x_b,grad_x_g,grad_x_r = cv2.split(grad_x)
        grad_x_sq_b = np.power(grad_x_b,2)
        grad_x_sq_g = np.power(grad_x_g,2)
        grad_x_sq_r = np.power(grad_x_r,2)
        
        grad_y = (takeYGradient(cvImage))
        grad_y_b,grad_y_g,grad_y_r = cv2.split(grad_y)
        grad_y_sq_b = np.power(grad_y_b,2)
        grad_y_sq_g = np.power(grad_y_g,2)
        grad_y_sq_r = np.power(grad_y_r,2)
        
        sqrt_b = np.power(np.add(grad_x_sq_b, grad_y_sq_b), .5)
        sqrt_g =np.power(np.add(grad_x_sq_g, grad_y_sq_g), .5)
        sqrt_r =np.power(np.add(grad_x_sq_r, grad_y_sq_r), .5)

        mg = cv2.merge((sqrt_b,sqrt_g,sqrt_r))
        return mg

    #add the two squared gradients together and take the square root.

#########################################################
###    Part B: k-Means Segmentation Functions
######################################################### 

# TODO:PA2 Fill in this function
def chooseRandomCenters(pixelList, k):
    """
    Choose k random starting point from a list of candidates.
    
    Parameters:
        pixelList - an (n x 6) matrix 
        
    Return:
        centers - a (k x 6) matrix composed of k random rows of pixelList
    """
    n,_=np.shape(pixelList)
    #none repeating set of indicies to choose
    indicies = np.random.choice(np.arange(n), k, replace = False) 
    samp = pixelList[indicies]
    return samp

# TODO:PA2 Fill in this function
def kMeansSolver(pixelList, k, centers=None, eps=0.001, maxIterations=100):
    '''
    Find a local optimum for the k-Means problem in 3D with
    Lloyd's Algorithm
    
    Assign the index of the center closest to each pixel
    to the fourth element of the row corresponding that 
    pixel.
    
    Parameters:
        pixelList - n x 6 matrix, where each row is <H, S, V, x, y, c>,
                    and c is index the center closest to it.
        centers - a k x 5 matrix where each row is one of the centers
                  in the k means algorithm. Each row is of the format
                  <H, S, V, x, y>
        eps - a positive real number user to test for convergence.
        maxIterations - a positive integer indicating how many 
                        iterations may occur before aborting.
    
    Return:
        iter - the number of iterations before convergence.
    '''
    # TODO:PA2 
    # H,S,V, x, and y values into the [0, 1] range.
    normalized_pixelList = np.zeros(np.shape(pixelList))
    normalized_pixelList[:,0:3] = np.true_divide(pixelList[:,0:3], 255)

    maxes = np.amax(pixelList[:,3:5], axis=0)
    max_xy = max(maxes[0],maxes[1])
    np.true_divide(pixelList[:,3:5], max_xy)
    # Initialize any data structures you need.
    newCenters = np.zeros((k,5))
    # END TODO:PA2
    
    if centers is None:
        centers = chooseRandomCenters(normalized_pixelList,k)[:,0:5]
    
    for iter in range(maxIterations):
        # TODO:PA2 Assign each point to the nearest center

        dists = scipy.spatial.distance.cdist(normalized_pixelList[:,0:5], centers, 'euclidean')#find the distance from each point to each center
        min_incies = np.argmin(dists, axis=1)#return the index of the min of each distance array
        normalized_pixelList[:,5] = min_incies #those indicies are the new c's
        pixelList[:,5] = min_incies #those indicies are the new c's

        # END TODO:PA2
        
        # TODO:PA2 Recalculate centers
        for c in range(k):#go through each center (0...k)
            centroid_members = normalized_pixelList[min_incies == c]    #separate out the pixels that are part of that center
            if (len(centroid_members)) == 0:#in a case where a given centroid has no members
               centroid_members = np.zeros((1,6))
            avgs = np.average(centroid_members[:,:5], axis=0)                 #average up each of their features which gives the new center
            newCenters[c] = avgs
            
        
        # END TODO:PA2
        
        validCenters = np.isfinite(newCenters)
        if (np.linalg.norm(centers[validCenters] - newCenters[validCenters], 2) < eps):
            centers = newCenters
            return iter
        else:
            centers = newCenters
    return iter 
      
def convertToHsv(rgbTuples):
    """
    Convert a n x 3 matrix whose rows are RGB tuples into
    an n x 3 matrix whose rows are the corresponding HSV tuples.
    The entries of rgbTuples should lie in [0,1]
    """
    B = rgbTuples[:,0]
    G = rgbTuples[:,1]
    R = rgbTuples[:,2]
    
    hsvTuples = np.zeros_like(rgbTuples)
    H = hsvTuples[:,0]
    S = hsvTuples[:,1]
    V = hsvTuples[:,2]
    
    alpha = 0.5 * (2*R - G - B)
    beta = np.sqrt(3)/2 * (G - B)
    H = np.arctan2(alpha, beta)
    V = np.max(rgbTuples,1)
    
    chroma = np.sqrt(np.square(alpha) + np.square(beta))
    S[V != 0] = np.divide(chroma[V != 0], V[V != 0])
    
    hsvTuples[:,0] = H  
    hsvTuples[:,1] = S  
    hsvTuples[:,2] = V  
    
    return hsvTuples
            
    
def kMeansSegmentation(cvImage, k, useHsv=True, eps=1e-14):
    """
    Execute a color-based k-means segmentation 
    """
    # Reshape the imput into a list of R,G,B,X,Y,C tuples, where
    # means that a pixel has not yet been assigned to a segment.
    m, n = cvImage.shape[0:2]
    numPix = m*n
    pixelList = np.zeros((numPix,6))
    pixelList[:,0:3] = cvImage.reshape((numPix,3))
    pixelList[:,3] = np.tile(np.arange(n),m)
    pixelList[:,4] = np.repeat(np.arange(m), n)
    
    # Convert the image to hsv.
    if useHsv:
        pixelList[:,:3] = convertToHsv(pixelList[:,:3]/255.)*255
    
    # Initialize k random centers in the color-position space.
    centers = (np.max(pixelList[:,0:5],0)-np.min(pixelList[:,0:5],0))*np.random.random((k,5))+np.min(pixelList[:,0:5],0)

    # Run Lloyd's algorithm until convergence
    iter = kMeansSolver(pixelList, k, eps=eps)
    
    # Color the pixels based on their centers
    if k <= 64:
        colors = np.array(COLORS[:k])
    else:
        colors = np.random.uniform(0,255,(k,3))
    
    R = pixelList[:,0]
    G = pixelList[:,1]
    B = pixelList[:,2]
    centerIndices = pixelList[:,5]
    
    for j in range(k):
       R[centerIndices == j] = colors[j,0] 
       G[centerIndices == j] = colors[j,1]
       B[centerIndices == j] = colors[j,2]
       
    return pixelList[:,:3].reshape(cvImage.shape).astype(np.uint8), iter
       
#########################################################
###    Part C: Normalized Cuts Segmentation Functions
######################################################### 

# TODO:PA2 Fill in this function
def getTotalNodeWeights(W):
    """
    Calculate the total weight of all edges leaving each 
    node.
    
    Parameters:
        W - the m*n x m*n weighted adjecency matrix of a graph
    
    Return:
        d - a vector whose ith component is the total weight
            leaving node i in W's graph.
    """
    m_x_n,_ = np.shape(W)
    d = np.zeros(m_x_n)
    sums = np.sum(W, axis=1)
    return sums

# TODO:PA2 Fill in this function
def approxNormalizedBisect(W, d):
    """
    Find the eigenvector approximation to the normalized cut
    segmentation problem with weight matrix W and diagonal d.
    
    Parameters:
        W - a (n*m x n*m) array of weights (floats)
        d - a n*m vector

    Return:
        y_1 - the second smallest eigenvector of D-W
    """
    m_n = len(d)
    I = np.identity(m_n)
    d_sqrt =  np.sqrt(d)
    neg_sqrt_d = np.reciprocal(d_sqrt)
    neg_sqrt_d_diag = np.diag(neg_sqrt_d)
    L = I-np.dot(np.dot((neg_sqrt_d_diag),W), neg_sqrt_d_diag)
    w,v = scipy.linalg.eigh(L)
    argsorted = np.argsort(w)
    z=v[:,argsorted[1]]
    d_inverse = np.reciprocal(d)
    return np.dot(neg_sqrt_d_diag, np.transpose(z))

# TODO:PA2 Fill in this function
def getColorWeights(cvImage, r, sigmaF=5, sigmaX=6):
    """
    Construct the matrix of the graph of the input image,
    where weights between pixels i, and j are defined
    by the exponential feature and distance terms.
    
    Parameters:
        cvImage - the m x n uint8 input image
        r - the maximum distance below which pixels are 
            considered to be connected
        sigmaF - the standard deviation of the feature term
        sigmaX - the standard deviation of the distance term
    
    Return:
        W - the m*n x m*n matrix of weights representing how 
            closely each pair of pixels is connected
    
    """
    sigmaXsq = sigmaX**2
    sigmaFsq = sigmaF**2
    shape = np.shape(cvImage)
    m,n = shape[0],shape[1]
    w = np.zeros((m*n,m*n))
    
    #o and p are the indicies of the pixel we are calculating for
    o=-1
    for i in range (m*n):
        p = i%n
        if i%n == 0:
            o+=1
        #k and l are the indicies of the pixel we are comparing to
        k = -1
        for j in range (m*n):
            l = j%n
            if j%n == 0:
                k+=1
            dist = np.linalg.norm(np.array([o,p])-np.array([k,l]))
            if dist <= r:
                x_exponent = (-1*dist)/sigmaXsq #distance between all j,k and i
                c_exponent = (-1*np.linalg.norm(cvImage[o][p]-cvImage[k][l]))/sigmaFsq
                entry   = np.multiply(np.exp(x_exponent), np.exp(c_exponent))
                w[i][j] = entry
    return w       

# TODO:PA2 Fill in this function
def reconstructNCutSegments(cvImage, y, threshold=0):
    """
    Create an output image that is yellow wherever y > threshold
    and blue wherever y < threshold
    
    Parameters:
        cvImage - an (m x n x 3) BGR image
        y - the (m x n)-dimensional eigenvector of the normalized 
            cut approximation algorithm,
        threshold - the cutoff between pixels in the two segments.
        
    Return:
        segmentedImage - an (n x m x 3) image that is yellows
                         for pixels with values above the threshold
                         and blue otherwise.
    """
    bools = y>threshold
    m,n,c = np.shape(cvImage)
    new = np.zeros((m,n,c))
    for i in range(m):
        for j in range(n):
            if bools[(i*n)+j]==True:
                new[i][j] = [0,255,255]
            else:
                new[i][j] = [255,0,0]
    return new
            
    
def nCutSegmentation(cvImage, sigmaF=5, sigmaX=6):
    print("Getting Weight Matrix")
    W = getColorWeights(cvImage, 7)
    print(str(W.shape[0]) + "x" + str(W.shape[1]) + " Weight matrix generated")
    d = getTotalNodeWeights(W)
    print("Calculated weight totals")
    y = approxNormalizedBisect(W, d)
    print("Reconstructing segments")
    segments = reconstructNCutSegments(cvImage, y, 0)
    return segments.astype(np.uint8)


    
    

    