# Please place imports here.
# BEGIN IMPORTS
import numpy as np
import cv2
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- 3 x N array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    
    #how do I avoid doing each pixel by itself
    #can I do inverse LT times I
    #print (np.shape(lights))
    #print (np.shape(np.transpose(lights)))
    #print (np.shape(np.dot(lights, np.transpose(lights))))
    
    shape_l = np.shape(lights)
    N,height,width,num_channels = np.shape(images)
    
    print ("i shape", (N,height,width,num_channels))
    
    rshp_images = np.reshape(images, (N,height*width*num_channels))
    
    print(np.shape(rshp_images))
    
    LLinv =  np.linalg.inv(np.dot(lights, np.transpose(lights)))
    LLinv_t_L = np.dot(LLinv, lights)
    print ("shape LLinv_t_L =", np.shape(LLinv_t_L))
    print ("shape rshp_images =", np.shape(rshp_images))
    G = np.dot(LLinv_t_L,rshp_images)
    #print("G shape", np.shape(G))
    #print("G", albedo)
    G3chan = np.reshape(G,(height,width,num_channels,3))
    Ggrayscale = np.mean(G3chan, axis=2)
    
    albedo = np.linalg.norm(Ggrayscale, axis = 2)
    
    
    print("albedo shape", np.shape(albedo))
    #print("albedo", albedo)
    
    
    #albedo_norm = np.linalg.norm(albedo, axis = 2)
    bools = albedo>1e-7
    #albedo=albedo*bools
   # print("albedo shape", np.shape(albedo))
   # print("albedo", albedo)
    normals = np.zeros(np.shape(albedo))
    x,y = np.shape(bools)
    for i in range(len(albedo)):
    # for i in range(x):
    #     for i in range(y):
        
        if bools[i] == False:
            normals[i,:] = np.zeros(3)
        else:
            normals[i,:] = np.divide(G[i,:], albedo)
    
    return albedo, normals

    # rImat = np.zeros((shape_l[1],shape_i[1]*shape_i[2]))
    # bImat = np.zeros((shape_l[1],shape_i[1]*shape_i[2]))
    # gImat = np.zeros((shape_l[1],shape_i[1]*shape_i[2]))
    
    rshp_images = np.reshape(shape_i[0],shape_i[1]*shape_i[2]*shape_i[3])
    
    # for num, pic in enumerate(images):
    #     if len(shape_i) == 4:
            
            # rpic = pic[:,:,0]
            # bpic = pic[:,:,1]
            # gpic = pic[:,:,2]
            # rflat = rpic.flatten()
            # bflat = bpic.flatten()
            # gflat = gpic.flatten()
            # rImat[num] = rflat
            # bImat[num] = bflat
            # gImat[num] = gflat
    
    
            
    # chan3Imat = np.zeros((shape_l[1],shape_i[1]*shape_i[2]),3)
    # chan3Imat[:,:,0] = rImat
    # chan3Imat[:,:,1] = bImat
    # chan3Imat[:,:,2] = gImat
    # meanImat = np.mean(chan3Imat ,axis = 2)
    # 
    # LLinv =  np.linalg.inv(np.dot(lights, np.transpose(lights)))
    # LLinv_t_L = np.dot(LLinv, lights)
    # rG = np.dot(rLLinv_t_L,rImat)
    # 
    # LLinv =  np.linalg.inv(np.dot(lights, np.transpose(lights)))
    # LLinv_t_L = np.dot(LLinv, lights)
    # bG = np.dot(bLLinv_t_L,bImat)
    # 
    # LLinv =  np.linalg.inv(np.dot(lights, np.transpose(lights)))
    # LLinv_t_L = np.dot(LLinv, lights)
    # gG = np.dot(bLLinv_t_L,bImat)


def pyrdown_impl(image):
    """
    Prefilters an image with a gaussian kernel and then downsamples the result
    by a factor of 2.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/16 [ 1 4 6 4 1 ]

    Functions such as cv2.GaussianBlur and
    scipy.ndimage.filters.gaussian_filter are prohibited.  You must implement
    the separable kernel.  However, you may use functions such as cv2.filter2D
    or scipy.ndimage.filters.correlate to do the actual
    correlation / convolution.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Downsampling should take the even-numbered coordinates with coordinates
    starting at 0.

    Input:
        image -- height x width [x channels] image of type float32.
    Output:
        down -- ceil(height/2) x ceil(width/2) [x channels] image of type
                float32.
    """
    #scipy.ndimage.filters.correlate()
    shape = np.shape(image)
    kern = np.array([.0625, .24, .375, .24, .0625])
    fltrd1 = cv2.filter2D(src=image, ddepth=-1, kernel=kern, borderType = cv2.BORDER_REFLECT_101)
    fltrd2 = cv2.filter2D(src=fltrd1, ddepth=-1, kernel=np.transpose(kern), borderType = cv2.BORDER_REFLECT_101)
    if len(shape) == 2:
        down = fltrd2[::2,::2]
    else: 
        down = fltrd2[::2,::2, :]
    #return down
    
    if len(shape) == 2:
        bad= image[::2,::2]
    else: 
        bad = image[::2,::2, :]
    return bad
    

def pyrup_impl(image):
    """
    Upsamples an image by a factor of 2 and then uses a gaussian kernel as a
    reconstruction filter.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/8 [ 1 4 6 4 1 ]
    Note: 1/8 is not a mistake.  The additional factor of 4 (applying this 1D
    kernel twice) scales the solution according to the 2x2 upsampling factor.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Upsampling should produce samples at even-numbered coordinates with
    coordinates starting at 0.

    Input:
        image -- height x width [x channels] image of type float32.
    Output:
        up -- 2 height x 2 width [x channels] image of type float32.
    """
    shape = np.shape(image)
    print ("shape is", shape)
    if len(shape) == 2:
        mixed = np.zeros((shape[0]*2,shape[1]*2))
        mixed[::2,::2] = image
    else:
        mixed = np.zeros((shape[0]*2,shape[1]*2, shape[2]))
        mixed[::2,::2, :] = image
    #
    kern = np.array([.125, .5, .75, .5, .125])
    fltrd1 = cv2.filter2D(src=mixed, ddepth=-1, kernel=kern, borderType = cv2.BORDER_REFLECT_101)
    fltrd2 = cv2.filter2D(src=fltrd1, ddepth=-1, kernel=np.transpose(kern), borderType = cv2.BORDER_REFLECT_101)
    
    return mixed
    
    
    
    #return fltrd2


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    
    #lec 17 slide 25
    
    #am I projecting from the real world to an image or from an image to the real world?
    
    #print ("shape K =", np.shape(K))
    #print ("shape Rt =", np.shape(Rt))
    P = np.dot(K, Rt)
    x,y,_ = np.shape(points)
    points_homo = np.concatenate((points,np.ones((x,y,1))),axis=2)
    print ("shape P =", np.shape(P))
    print ("shape points_homo =", np.shape(points_homo))
    #print ("points", points)
    #projections = np.dot(P,points_homo) #do I invert P?
    projections = np.tensordot(points_homo, np.transpose(P), axes = 1)

    return projections
    
    #shape = np.shape(points)
    #return np,zeros((shape[0],shape[1],2))
    
def unproject_corners_impl(K, width, height, depth, Rt):
    """
    Undo camera projection given a calibrated camera and the depth for each
    corner of an image.

    The output points array is a 2x2x3 array arranged for these image
    coordinates in this order:

     (0, 0)      |  (width, 0)
    -------------+------------------
     (0, height) |  (width, height)

    Each of these contains the 3 vector for the corner's corresponding
    point in 3D.

    Tutorial:
      Say you would like to unproject the pixel at coordinate (x, y)
      onto a plane at depth z with camera intrinsics K and camera
      extrinsics Rt.

      (1) Convert the coordinates from homogeneous image space pixel
          coordinates (2D) to a local camera direction (3D):
          (x', y', 1) = K^-1 * (x, y, 1)
      (2) This vector can also be interpreted as a point with depth 1 from
          the camera center.  Multiply it by z to get the point at depth z
          from the camera center.
          (z * x', z * y', z) = z * (x', y', 1)
      (3) Use the inverse of the extrinsics matrix, Rt, to move this point
          from the local camera coordinate system to a world space
          coordinate.
          Note:
            | R t |^-1 = | R' -R't |
            | 0 1 |      | 0   1   |

          p = R' * (z * x', z * y', z * 1)' - R't

    Input:
        K -- camera intrinsics calibration matrix
        width -- camera width
        height -- camera height
        depth -- depth of plane with respect to camera
        Rt -- 3 x 4 camera extrinsics calibration matrix
    Output:
        points -- 2 x 2 x 3 array of 3D points
    """
    Kinv = np.linalg.inv(K)
    m4x3 = np.zeros((4,3))
    m4x3[0] = np.multiply(depth, np.dot(Kinv, [0,0,1]))
    m4x3[1] = np.multiply(depth, np.dot(Kinv, [width,0,1]))
    m4x3[2] = np.multiply(depth, np.dot(Kinv, [0,height,1]))
    m4x3[3] = np.multiply(depth, np.dot(Kinv, [width,height,1]))
    #m2x2x3 = np.reshape(m4x3,(2,2,3))
    
    tpose_R = Rt[0:3, 0:3]
    #Rinv = np.zeros((4,4))
    tpose_R_t_t = np.dot(tpose_R, Rt[:,3:])
    #Rinv[0:3, 0:3] = tpose_R
    #handles the different ps all at once with a matrix multiplication
    #do I need to transpose
    print (Rt)
    print (tpose_R)
    print (Rt[:,3:])
    print(tpose_R_t_t)
    print ("shape tpose_R_t_t", tpose_R_t_t)
    print ("shape tpose_R, np.transpose(m4x3))", tpose_R, np.transpose(m4x3))
    p4x3 = np.dot(tpose_R, np.transpose(m4x3)) - tpose_R_t_t#add back in after
    p2x2x3 = np.reshape(p4x3,(2,2,3))
    return p2x2x3
    #return np.zeros((2,2,3))

def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the channel-interleaved, column
    major order (more simply, flatten on the transposed patch).
    For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x112, x211, x212, x121, x122, x221, x222 ]

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    #should I padd it first
    
    #convolve with a mean filter
    

    x,y,num_chan= np.shape(image)
    # final_mat = np.zeros((x,y,num_chan*ncc_size**2))
    # 
    # for c in range (num_chan):
    #     single_chan_image = image[:,:, c]
    #     assert np.shape(single_chan_image) == (x,y)
    #     mean_subracted_mat = np.zeros((x,y))
    #     for i in range(ncc_size//2, x-ncc_size//2):
    #         for j in range(ncc_size//2, y-ncc_size//2):
    #             mean = np.mean(single_chan_image[i:i+ncc_size, j:j+ncc_size])
    #             mean_subracted_mat[:,:, single_chan_image] = single_chan_image[i:i+ncc_size, j:j+ncc_size] - mean
    # num = 0                 
    # for i in range(x):
    #     for j in range(y):
    #         
    #         if i<ncc_size//2 or i>=x-ncc_size//2:
    #            new_patch = np.zeros((ncc_size,ncc_size)) 
    #         else:
    #             patch = mean_subracted_mat[i:i+ncc_size, j:j+ncc_size, :]
    #             norm = np.linalg.norm(patch)
    #             if norm < 1e-6: 
    #                new_patch = patch*0
    #             else:
    #                 new_patch = patch/norm
    #         patch_vec = np.flatten(np.transpose(new_patch))
    #         num += 1
    #         final_mat[x,y,num]
    # #return final_mat
    return np.zeros((x,y,num_chan*ncc_size**2))
    
def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    return np.sum(np.multiply(image1, image2), axis = 2)
    