import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches
import os


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    pano_im_W = im1.shape[1] + im2.shape[1]
    pano_im_H = im2.shape[0]
    im1_H, im1_W = im1.shape[:2]
    pano_im = cv2.warpPerspective(im2, H2to1, (pano_im_W, pano_im_H))
    cv2.imwrite('warp.jpg',pano_im)
    for x in range(im1_W):
        for y in range(im1_H):
            '''
            if pano_im[y,x].sum() < warp_im1[y,x].sum():
                pano_im[y,x] = warp_mask1[y,x] * warp_im1[y,x] + (1-warp_mask1[y,x]) * warp_im2[y,x]
            '''
            if pano_im[y,x].sum() < im1[y,x].sum():
                pano_im[y,x] = im1[y,x]
    #pano_im[0:im1.shape[0], 0:im1.shape[1]] = im1

    #mask = np.zeros((im1.shape[0], im1.shape[1]))
    '''
    mask = np.zeros((6, 6))
    mask[0,:] = 1
    mask[-1,:] = 1
    mask[:,0] = 1
    mask[:,-1] = 1
    mask = distance_transform_edt(1-mask)
    mask = mask/(mask.max(0) + 0.00001)
 
    print(mask)
    '''
    #cv2.imwrite('mask.jpg', mask)

    cv2.imwrite('pano_im.jpg',pano_im)

    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix without clipping. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    im2_H, im2_W = im2.shape[:2]
    us = np.array([[0,0,1], [0, im2_H-1,1], [im2_W-1, 0, 1], [im2_W-1,im2_H-1,1]])

    x_min = 0
    y_min = 0
    x_max = im2_W
    y_max = im2_H
    for u in us:
        x = H2to1.dot(u)
        x /= x[-1]
        #print(x)
        if x_min > x[0]:
            x_min = x[0]
        if x_max < x[0]:
            x_max = x[0]
        if y_min > x[1]:
            y_min = x[1]
        if y_max < x[1]:
            y_max = x[1]



    margin = 0
    pano_im_W = int(x_max - x_min + margin)
    pano_im_H = int(y_max - y_min + margin)

    translation = np.dot(H2to1, np.array([0,0,1]))
    translation /= translation[-1]
    M = np.identity(3)

    M[0, -1] = abs(x_min)
    M[1, -1] = abs(y_min)

    warp_im1 = cv2.warpPerspective(im1, M, (pano_im_W, pano_im_H))
    warp_im2 = cv2.warpPerspective(im2, np.dot(M, H2to1), (pano_im_W, pano_im_H))

    #cv2.imwrite('warp_im1.jpg',warp_im1)
    #cv2.imwrite('warp_im2.jpg',warp_im2)

    pano_im = warp_im2

    '''
    mask1 = np.zeros((im1.shape[0], im1.shape[1]))
    mask1[0,:] = 1
    mask1[-1,:] = 1
    mask1[:,0] = 1
    mask1[:,-1] = 1
    mask1 = distance_transform_edt(1-mask1)
    mask1 = mask1/(mask1.max(0) + 0.00001)
    
    mask2 = np.zeros((im2.shape[0], im2.shape[1]))
    mask2[0,:] = 1
    mask2[-1,:] = 1
    mask2[:,0] = 1
    mask2[:,-1] = 1
    mask2 = distance_transform_edt(1-mask2)
    mask2 = mask2/(mask2.max(0) + 0.00001)
    
    warp_mask1 = cv2.warpPerspective(mask1, M, (pano_im_W, pano_im_H))
    warp_mask2 = cv2.warpPerspective(mask2, np.dot(M, H2to1), (pano_im_W, pano_im_H))
    '''
    for x in range(pano_im_W):
        for y in range(pano_im_H):
            '''
            if pano_im[y,x].sum() < warp_im1[y,x].sum():
                pano_im[y,x] = warp_mask1[y,x] * warp_im1[y,x] + (1-warp_mask1[y,x]) * warp_im2[y,x]
            '''
            if pano_im[y,x].sum() < warp_im1[y,x].sum():
                pano_im[y,x] = warp_im1[y,x]
            

    cv2.imwrite('pano_im_noClip.jpg', pano_im)
    return pano_im


def generatePanaroma(im1, im2):
    '''
    Generate and save panorama of im1 and im2.

    INPUT
        im1 and im2 - two images for stitching
    OUTPUT
        Blends img1 and warped img2 (with no clipping) 
        and saves the panorama image.
    '''

    
    # load test pattern for Brief
    H2to1_file = '../results/q6_1.npy'
    if os.path.isfile(H2to1_file):
        # load from file if exists
        H2to1 = np.load(H2to1_file)
    else:
        # produce and save patterns if not exist
        locs1, desc1 = briefLite(im1)
        locs2, desc2 = briefLite(im2)
        matches = briefMatch(desc1, desc2)
        H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
        
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(H2to1_file, H2to1)

    pano_im = imageStitching_noClip(im1, im2, H2to1)
    #pano_im = imageStitching(im1, im2, H2to1)


    return pano_im





if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    '''
    #im1 = cv2.imread('../data/chickenbroth_02.jpg')
    #im2 = cv2.imread('../data/chickenbroth_03.jpg')

    scale_percent = 20 # percent of original size
    width = int(im1.shape[1] * scale_percent / 100)
    height = int(im1.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    scale_im1 = cv2.resize(im1, dim)
    
    width = int(im2.shape[1] * scale_percent / 100)
    height = int(im2.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    scale_im2 = cv2.resize(im2, dim)

    #generatePanaroma(scale_im1, scale_im2)
    '''
    generatePanaroma(im1, im2)