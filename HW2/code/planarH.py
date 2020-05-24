import numpy as np
import cv2
from BRIEF import briefLite, briefMatch


def computeH(p1, p2):
    '''
    INPUTS
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                coordinates between two images
    OUTPUTS
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
                equation
    '''
    
    #print(p1.shape)
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    #let p1 = x  p2 = u

    A = []
    N = p1.shape[1]
    #N = 4
    #print(p1)
    #print(p2)
    for i in range(N):
        xi = p1[0,i]
        yi = p1[1,i]
        ui = p2[0,i]
        vi = p2[1,i]
        A.append([0,0,0,-ui,-vi,-1,yi*ui,yi*vi,yi])
        A.append([ui,vi,1,0,0,0,-xi*ui,-xi*vi,-xi])

    U,S,Vt = np.linalg.svd(A)
    H2to1 = Vt[-1, :].reshape((3,3))
    H2to1 /= H2to1[2,2]

    return H2to1


def ransacH(matches, locs1, locs2, num_iter, tol):
    '''
    Returns the best homography by computing the best set of matches using RANSAC
    
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches         - matrix specifying matches between these two sets of point locations
        nIter           - number of iterations to run RANSAC
        tol             - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''

    ###########################
    # TO DO ...
    num_sample_point = 8

    bestH = np.array([])
    max_match_cnt = 0

    for i in range(num_iter):
        indices = np.random.randint(len(matches), size=num_sample_point)
        #print(indices)
        sup_p1 = []
        sup_p2 = []
        for index in indices:
            idx1, idx2 = matches[index]
            sup_p1.append(locs1[idx1,:2])
            sup_p2.append(locs2[idx2,:2])
        H = computeH(np.array(sup_p1).T, np.array(sup_p2).T)
        #x = Hu
        match_cnt = 0
        for idx1, idx2 in matches:
            x = locs1[idx1].copy()
            u = locs2[idx2].copy()
            x[-1] = 1
            u[-1] = 1
            predict_x = H.dot(u)
            predict_x /= (predict_x[-1] + 0.00001)
            err = x - predict_x
            err_val = np.linalg.norm(err[:2])
            if err_val <= tol:
                match_cnt += 1
        if max_match_cnt < match_cnt:
            max_match_cnt = match_cnt
            bestH = H
        print('number_iter: ' + str(i) + ' max_match_cnt = ' + str(max_match_cnt))
    
    acc = max_match_cnt / len(matches)
    print(acc)

    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    #im1 = cv2.imread('../data/chickenbroth_01.jpg')
    #im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)

    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)