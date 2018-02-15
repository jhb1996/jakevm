import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
	# TODO-BLOCK-BEGIN
    imgShape = np.shape(img)
    imgNumRows = imgShape[0] 
    imgNumCols = imgShape[1] 
    kerShape = np.shape(kernel)
    kerNumRows = kerShape[0] 
    kerNumCols = kerShape[1] 
    ccdMat = np.zeros(imgShape)
    grayscale = (len(imgShape)==2)
    if (grayscale):
    	padded = np.zeros((imgNumRows+kerNumRows+1, imgNumCols+kerNumCols+1))
    	padded[kerNumRows/2+1:imgNumRows+kerNumRows/2+1, kerNumCols/2+1:imgNumCols+kerNumCols/2+1] = img
    	for i in range (imgNumRows):
	    	for j in range (imgNumCols):
	    		r1 = i+1
	    		r2 = r1 + kerNumRows
	    		c1 = j+1
	    		c2 = c1 + kerNumCols
	    		piece = padded[r1:r2, c1:c2]
	    		mtp = np.multiply(piece, kernel)
	    		sm = np.sum(mtp)
	    		ccdMat[i][j] = sm
	return ccdMat
    else:
    	padded = np.zeros((imgNumRows+kerNumRows+1, imgNumCols+kerNumCols+1, 3))

    	padded[kerNumRows/2+1:imgNumRows+kerNumRows/2+1, kerNumCols/2+1:imgNumCols+kerNumCols/2+1, :] = img
    	#extend the kernal into (wxhx3)
        k3D = np.zeros((kerNumRows,kerNumCols,3))
    	k3D[:,:,0] = kernel
    	k3D[:,:,1] = kernel
    	k3D[:,:,2] = kernel
    	for i in range (imgNumRows):
	    	for j in range (imgNumCols):
	    		r1 = i+1
	    		r2 = r1 + kerNumRows
	    		c1 = j+1
	    		c2 = c1 + kerNumCols
	    		piece = padded[r1:r2, c1:c2, :]
	    		mtp = np.multiply(piece, k3D)
	    		sm1 = np.sum(mtp, axis = 0)
	    		sm2 = np.sum(sm1, axis = 0)
	    		ccdMat[i][j] = sm2
    	return ccdMat

    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = np.flipud(np.fliplr(kernel)) #flip it once in either direction
    return cross_correlation_2d(img, kernel)
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, width, height):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    # asked a TA and he said I do not need to be able to handle even gaussians
    gaus_mat = np.zeros((width,height))
    for i in range (width):
    	for j in range (height):
    		y = i -(width/2.)+.5
    		x = j -(height/2.)+.5
    		gaus_mat[i][j] = (2.71828)**((-1*(((x)**2)+((y)**2)))/(2.*((float(sigma))**2))) #why does math.e not work

    return gaus_mat/(np.sum(gaus_mat))
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    gaus_mat = gaussian_blur_kernel_2d(sigma,size,size)
    filtered = cross_correlation_2d(img, gaus_mat)
    return filtered
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    
    # TODO-BLOCK-BEGIN
    return img-low_pass(img, sigma, size)
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
