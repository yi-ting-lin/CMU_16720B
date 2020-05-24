import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector


def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF
    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
        patch_width - the width of the image patch (usually 9)
        nbits       - the number of tests n in the BRIEF descriptor

    OUTPUTS
        compareX and compareY - LINEAR indices into the patch_width x patch_width image 
                                patch and are each (nbits,) vectors. 
    '''
    
    compareX = []
    compareY = []

    for i in range(nbits):
        lower_bound = -patch_width//2 + 1
        upper_bound = patch_width//2 + 1
        compareX.append(np.random.randint(lower_bound,upper_bound,2))
        compareY.append(np.random.randint(lower_bound,upper_bound,2))

    compareX = np.array(compareX)
    compareY = np.array(compareY)

    return  compareX, compareY


# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])

'''
for i in range(len(compareX)):
    x = [compareX[i, 0], compareY[i, 0]]
    y = [compareX[i, 1], compareY[i, 1]]
    plt.plot(x, y, 'r-')
plt.show()
'''

def computeBrief(im, gaussian_pyramid, locsDoG, k, levels,
    compareX, compareY):
    '''
    Compute brief feature
    INPUT
        locsDoG - locsDoG are the keypoint locations returned by the DoG
                detector.
        levels  - Gaussian scale levels that were given in Section1.
        compareX and compareY - linear indices into the 
                                (patch_width x patch_width) image patch and are
                                each (nbits,) vectors.
    
    
    OUTPUT
        locs - an m x 3 vector, where the first two columns are the image
                coordinates of keypoints and the third column is the pyramid
                level of the keypoints.
        desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
                of valid descriptors in the image and will vary.
    '''
    
    ##############################
    # TO DO ...
    # compute locs, desc here
    H = im.shape[0]
    W = im.shape[1]
    n_bits = len(compareX)
    #print(n_bits)
    locs = []
    desc = []

    #convert into grayscale
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #print(im.shape)


    #print(locsDoG.shape)

    for i in range(len(locsDoG)):
        x = locsDoG[i, 0]
        y = locsDoG[i, 1]
        if x < 4 or x > W - 5:
            continue
        if y < 4 or y > H - 5:
            continue
        locs.append(locsDoG[i])
        encode_set = []
        for bit in range(n_bits):
            p1 = [x + compareX[bit, 0], y + compareY[bit, 0]]
            p2 = [x + compareX[bit, 1], y + compareY[bit, 1]]
            if im[p1[1], p1[0]] < im[p2[1], p2[0]]:
                encode_set.append(1)
            else:
                encode_set.append(0)
        desc.append(encode_set)
                
    locs = np.array(locs)
    desc = np.array(desc)
    
    #print(locs.shape)
    #print(desc.shape)

    return locs, desc


def briefLite(im):
    '''
    INPUTS
        im - gray image with values between 0 and 1

    OUTPUTS
        locs - an m x 3 vector, where the first two columns are the image coordinates 
            of keypoints and the third column is the pyramid level of the keypoints
        desc - an m x n bits matrix of stacked BRIEF descriptors. 
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''
    
    #compute gaussian_pyramid and get key points
    locsDoG, gaussian_pyramid = DoGdetector(im)

    #load test pattern for Brief
    test_pattern_file = '../results/testPattern.npy'
    compareX, compareY = np.load(test_pattern_file)

    #compute brief
    locs, desc = computeBrief(im, gaussian_pyramid, locsDoG, 0, [], compareX, compareY)

    #print(locs)
    return locs, desc


def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    INPUTS
        desc1, desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    OUTPUTS
        matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.show()    
    

if __name__ == '__main__':
    # test makeTestPattern
    #compareX, compareY = makeTestPattern()
    
    # test briefLite
    
    #im = cv2.imread('../data/model_chickenbroth.jpg')
    '''
    im = cv2.imread('../data/chickenbroth_01.jpg')
    locs, desc = briefLite(im)
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.plot(locs[:,0], locs[:,1], 'r.')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)
    '''
    # test matches
   
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    #im1 = cv2.imread('../data/pf_scan_scaled.jpg')
    #im2 = cv2.imread('../data/pf_desk.jpg')
    '''
    if im1.shape[2] == 1:
        im1 = np.stack((im1,)*3, axis=-1)
    if im2.shape[2] == 1:
        im2 = np.stack((im2,)*3, axis=-1)
    '''
    '''
    scale_percent = 100 # percent of original size
    width = int(im1.shape[1] * scale_percent / 100)
    height = int(im1.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    im1 = cv2.resize(im1, dim)
    
    width = int(im2.shape[1] * scale_percent / 100)
    height = int(im2.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    im2 = cv2.resize(im2, dim)


    '''

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)

    #plotMatches(im1,im2,matches,locs1,locs2)
    