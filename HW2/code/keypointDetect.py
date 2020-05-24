import numpy as np
import cv2


def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    #print(im_pyramid.shape)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    INPUTS
        gaussian_pyramid - A matrix of grayscale images of size
                            [imH, imW, len(levels)]
        levels           - the levels of the pyramid where the blur at each level is
                            outputs

    OUTPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
        DoG_levels  - all but the very first item from the levels vector
    '''
    
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    
    DoG_levels = levels[1:]
    #print(DoG_levels)
    for i in range(1, len(levels)):
        Dog_result = gaussian_pyramid[:,:,i] - gaussian_pyramid[:,:,i-1]
        DoG_pyramid.append(Dog_result)
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    #print(DoG_pyramid.shape)
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = []
    

    imH, imW, levels = DoG_pyramid.shape
     
    epison = 0.00001
    
    for i in range(levels):
        Dx = cv2.Sobel(DoG_pyramid[:,:,i], cv2.CV_32F, dx=1, dy=0)
        Dy = cv2.Sobel(DoG_pyramid[:,:,i], cv2.CV_32F, dx=0, dy=1)
        Dxx = cv2.Sobel(Dx, cv2.CV_32F, dx=1, dy=0)
        Dxy = cv2.Sobel(Dx, cv2.CV_32F, dx=0, dy=1)
        Dyx = cv2.Sobel(Dy, cv2.CV_32F, dx=1, dy=0)
        Dyy = cv2.Sobel(Dy, cv2.CV_32F, dx=0, dy=1)
        R = np.zeros((imH, imW))
        for row in range(imH):
            for col in range(imW):
                H = np.array([[Dxx[row,col], Dxy[row,col]], [Dyx[row, col], Dyy[row,col]]])
                R[row, col] = (np.trace(H)**2 + 0.00001) / (np.linalg.det(H) + epison)
                
        principal_curvature.append(R)
    
    principal_curvature = np.stack(principal_curvature, axis=-1)
    #print(principal_curvature.shape)

    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = []
    imH, imW = DoG_pyramid.shape[:2]

    #print(imH)
    #print(imW)
    ##############
    #  TO DO ...
    # Compute locsDoG here
    dxs = [1,0,-1]
    dys = [1,0,-1]
    for i in range(len(DoG_levels)):
        for row in range(1, imH-1):
            for col in range(1, imW-1):
                cur_pixel = DoG_pyramid[row,col,i]
                cur_is_max = True
                for dx in dxs:
                    for dy in dys:
                        if cur_is_max == True and DoG_pyramid[row+dx, col+dy,i] > cur_pixel:
                            cur_is_max = False
                            
                if cur_is_max == False:
                    continue
                if i != 0 and cur_pixel < DoG_pyramid[row, col,i-1]:
                    continue
                if i != len(DoG_levels)-1 and cur_pixel < DoG_pyramid[row, col,i+1]:
                    continue
                if abs(cur_pixel) <= th_contrast:
                    continue
                if principal_curvature[row,col,i] > th_r:
                    continue
                locsDoG.append(np.array([col,row,i])) 
        
    locsDoG = np.stack(locsDoG, axis=0)
    #print(locsDoG.shape)
    return locsDoG
  

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    INPUTS          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.


    OUTPUTS         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, locsDoG here

    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    #print(locsDoG.shape)
   

    # Displaying the image  
    '''
    [cv2.circle(im, (locsDoG[i,0], locsDoG[i,1]), 1, (0, 255, 0),-1) for i in range(locsDoG.shape[0])]
    cv2.imshow('Image', im)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    #im_pyr = createGaussianPyramid(im)
    #displayPyramid(im_pyr)
    
    # test DoG pyramid
    #DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    #displayPyramid(DoG_pyr)
    
    # test compute principal curvature
    #pc_curvature = computePrincipalCurvature(DoG_pyr)
    
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    #locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)