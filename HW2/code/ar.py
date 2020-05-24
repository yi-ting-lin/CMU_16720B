import numpy as np
import cv2
import os
from planarH import computeH
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv

def compute_extrinsics(K, H):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        H - estimated homography
    OUTPUTS:
        R - relative 3D rotation
        t - relative 3D translation
    '''
    #calculate H'
    inv_K = np.linalg.inv(K)
    H_p = np.dot(inv_K, H)

    #calcluate the svd of H'[:,:2]
    U,S,Vt = np.linalg.svd(H_p[:,:2])

    #get R[:,:2]
    R = np.zeros((3,3))
    R[:,:2] = np.dot(np.dot(U, np.array([[1,0],[0,1],[0,0]])), Vt)
    
    #print(R)
    #get R[:,-1]
    R[:,-1] = np.cross(R[:,0], R[:,1])
    #print(np.linalg.det(R))
    if np.linalg.det(R) == -1:
        R[:,-1] *= -1
    #get t
    lamda_p = 0
    for m in range(3):
        for n in range(2):
            lamda_p += (H_p[m,n] / R[m,n])
    
    lamda_p /= 6
    t = H_p[:,-1] / lamda_p

    return R, t


def project_extrinsics(K, W, R, t):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        W - 3D planar points of textbook
        R - relative 3D rotation
        t - relative 3D translation
    OUTPUTS:
        X - computed projected points
    '''

    #############################
    # TO DO ...
    R_t = np.hstack((R,t.reshape(3,1)))
    W = np.vstack((W, np.ones(W.shape[1])))
    X = np.dot(np.dot(K,R_t), W)
    for col in range(X.shape[1]):
        X[:,col] /= (X[-1,col])
    return X


if __name__ == "__main__":
    # image
    #im = cv2.imread('../data/prince_book.jpeg')

    #define matrix
    K = np.array([[3043.72,0.0,1196.00],
                  [0.0,3043.72,1604.00],
                  [0.0,0.0,1.0]])
    W = np.array([[0.0,18.2,18.2,0.0],
                  [0.0,0.0,26.0,26.0],
                  [0.0,0.0,0.0,0.0]])
    
    Ｘ = np.array([[483,1704,2175,67],
                  [810,781,2217,2286]])
    

    H = computeH(X,W[:2])
    R,t = compute_extrinsics(K,H)

    Wn = []
    
    with open('../data/sphere.txt') as file:
        reader = csv.reader(file, delimiter = ' ')
        for line in reader:
            for num in line:
                if num != '':
                    Wn.append(float(num))

    
    Wn = np.array(Wn).reshape((3,len(Wn)//3))
    Xn = project_extrinsics(K, Wn, R, t)

    Xn = Xn[:2,:]
    x_points = Xn[0,:] + 320
    y_points = Xn[1,:] + 630
    img=mpimg.imread('../data/prince_book.jpeg')
    imgplot = plt.imshow(img)
    plt.plot(x_points,y_points,linewidth=0.5)
    plt.savefig('ar_plot.jpg')
    