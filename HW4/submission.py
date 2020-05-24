"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import sys
import helper
import scipy.optimize
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    T = np.array([[1.0/M, 0, 0],[0, 1.0/M, 0], [0,0,1.0]])
    
    #Normalize the coordinate and get x1, y1, x2, y2
    pts1 = pts1.astype(float)/M
    pts2 = pts2.astype(float)/M
    
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    

    #Get A matrix
    N = x1.shape[0]
    A = np.ones((N, 9)).astype(float)
    A[:, 0] = (x2*x1).T
    A[:, 1] = (x2*y1).T
    A[:, 2] = x2.T
    A[:, 3] = (y2*x1).T
    A[:, 4] = (y2*y1).T
    A[:, 5] = y2.T
    A[:, 6] = x1.T
    A[:, 7] = y1.T

    #Get F from the SVD decomposition
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1,:].reshape(3,3)

    #Refine & singularize F
    F = helper.refineF(F, pts1, pts2)

    #unscale F
    F = np.dot(T.T, F).dot(T)
    #print('F = ')
    #print(F)
    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''

def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    T = np.array([[1.0/M, 0, 0],[0, 1.0/M, 0], [0,0,1.0]])
    
    #Normalize the coordinate and get x1, y1, x2, y2
    pts1 = pts1.astype(float)/M
    pts2 = pts2.astype(float)/M
    
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    

    #Get A matrix
    N = x1.shape[0]
    A = np.ones((N, 9)).astype(float)
    A[:, 0] = (x2*x1).T
    A[:, 1] = (x2*y1).T
    A[:, 2] = x2.T
    A[:, 3] = (y2*x1).T
    A[:, 4] = (y2*y1).T
    A[:, 5] = y2.T
    A[:, 6] = x1.T
    A[:, 7] = y1.T

    #Get F from the SVD decomposition
    _, _, Vt = np.linalg.svd(A)
    F1 = Vt[-1,:].reshape(3,3)
    F2 = Vt[-2,:].reshape(3,3)


    #Estimate polynomial: a0 + a1x + a2x^2 + a3x^3 = func(x)
    func = lambda x: np.linalg.det(x * F1 + (1 - x) * F2)
    a0 = func(0)
    a1 = 2*(func(1)-func(-1))/3 - (func(2)-func(-2))/12
    a2 = (func(1)+func(-1))/2 - a0
    a3 = (func(1)-func(-1))/2 - a1


    #Get roots and Farray
    poly_roots = np.roots([a3, a2, a1, a0])
    Farray = [r*F1 + (1.0-r)*F2 for r in poly_roots]
    
    #Refine & singularize F
    #Farray = [helper.refineF(F, pts1, pts2) for F in Farray]

    #unscale F
    Farray = [np.dot(T.T, F).dot(T) for F in Farray]

    #Print F in Farray
    #for F in Farray:
    #    print(F)  

    return Farray


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = np.dot(K2.T, F).dot(K1)
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    x1i = pts1[:,0]
    y1i = pts1[:,1]
    x2i = pts2[:,0]
    y2i = pts2[:,1]

    #P: Nx3 
    P = []
    for x1,y1,x2,y2 in zip(x1i,y1i,x2i,y2i):
        A = np.array([[C1[0,0]-C1[2,0]*x1,C1[0,1]-C1[2,1]*x1,C1[0,2]-C1[2,2]*x1,C1[0,3]-C1[2,3]*x1],
                      [C1[1,0]-C1[2,0]*y1,C1[1,1]-C1[2,1]*y1,C1[1,2]-C1[2,2]*y1,C1[1,3]-C1[2,3]*y1],
                      [C2[0,0]-C2[2,0]*x2,C2[0,1]-C2[2,1]*x2,C2[0,2]-C2[2,2]*x2,C2[0,3]-C2[2,3]*x2],
                      [C2[1,0]-C2[2,0]*y2,C2[1,1]-C2[2,1]*y2,C2[1,2]-C2[2,2]*y2,C2[1,3]-C2[2,3]*y2]])
        _, _, Vt = np.linalg.svd(A)
        p = Vt[-1,:]
        p /= p[-1]
        P.append(p[:3])
    P = np.array(P)

    wi = np.ones(P.shape[1]+1)
    err = 0

    for x1,x2,p in zip(pts1,pts2,P):
        wi[:3] = p
        #re-projection: proj_x1 = dot(C1, wi) proj_x2 = dot(C2, wi)  proj_x1 & proj_x2: 3x1
        proj_x1 = np.dot(C1, wi.reshape(wi.shape[0], 1))
        proj_x2 = np.dot(C2, wi.reshape(wi.shape[0], 1))

        #Normalization
        proj_x1 /= proj_x1[-1]
        proj_x2 /= proj_x2[-1]

        #Get the first two elements into 2x1 and transpose to 1x2 for error calculation
        proj_x1 = proj_x1[:2].T
        proj_x2 = proj_x2[:2].T
        err += ((proj_x1-x1) **2 + (proj_x2-x2)**2).sum()
    
    return P, err

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    sigma = 5
    ker_size = 11
    #Generate gaussain mask (ref: https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy)
    weight_mask = gkern(ker_size, sigma)

    if len(im1.shape) > 2:
        weight_mask_stack = np.zeros((weight_mask.shape[0], weight_mask.shape[1], 3))
        weight_mask_stack[:,:,0] = weight_mask.copy()
        weight_mask_stack[:,:,1] = weight_mask.copy()
        weight_mask_stack[:,:,2] = weight_mask.copy()
        weight_mask = weight_mask_stack
    
    #Get epipolar line from F & pixel: l = dot(F, x)  l=[a,b,c] -> ax + by + c = 0
    l = np.dot(F, np.array([[x1], [y1], [1]]))
    
    #Get the patch of im1 at center (x1, y1) 
    offset = ker_size//2
    patch1 = im1[y1-offset:y1+offset+1, x1-offset:x1+offset+1]

    #Get points on the epipolar line of im2: X2, Y2
    H = im2.shape[0]
    W = im2.shape[1]
    search_offset = 30
    #X2 = np.array(range(x1-search_offset, x1+search_offset))
    #Y2 = np.round(-(l[0]*X2+l[2])/l[1]).astype(int)
    Y2 = np.array(range(y1-search_offset, y1+search_offset))
    X2 = np.round(-(l[1]*Y2+l[2])/l[0]).astype(int)
    valid = (X2 >= offset) & (X2 < W - offset) & (Y2 >= offset) & (Y2 < H - offset)
    X2 = X2[valid]
    Y2 = Y2[valid]

    #Find correspondance: x2 y2
    x2 = None
    y2 = None
    min_diff = sys.maxsize
    for x2_local, y2_local in zip(X2, Y2):
        patch2 = im2[y2_local-offset:y2_local+offset+1, x2_local-offset:x2_local+offset+1]
        diff = ((patch1 - patch2)**2 * weight_mask).sum()
        if diff < min_diff:
            x2 = x2_local
            y2 = y2_local
            min_diff = diff

    return x2, y2

def gkern(ker_size, sigma):
    ax = np.linspace(-(ker_size - 1) / 2., (ker_size - 1) / 2., ker_size)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    return kernel / np.sum(kernel)

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''


def ransacF(pts1, pts2, M):

    max_iter = 6000
    N = pts1.shape[0]
    threshold = 2
    max_inliers_cnt = 0
    #F = None
    inliers = None
    
    for i in range(max_iter):
        #Randomly pick 7 points from pts1
        #np.random.seed(i*2)
        indices = np.random.randint(N, size=7)
        #print(indices)
        #estimate F using 7 point algorithm
        F7s = sevenpoint(pts1[indices], pts2[indices], M)
        

        for F7 in F7s:
            #calculate epipolar lines l2s on im2 with respect to pts1 on im1 l: 3xN
            l1s = np.dot(np.hstack((pts2, np.ones((N,1)))), F7).T
            
            # get l:ax + by +c = 0
            l1_a_coes = l1s[0,:]
            l1_b_coes = l1s[1,:]
            l1_c_coes = l1s[2,:]
            #calculate dists
            dists1 = abs(l1_a_coes*pts1[:, 0] + l1_b_coes*pts1[:,1] + l1_c_coes)
            dists1 = dists1/np.sqrt(l1_a_coes**2 + l1_b_coes**2)
            
            #calculate epipolar lines l2s on im2 with respect to pts1 on im1 l: 3xN
            l2s = np.dot(F7, np.vstack((pts1.T, np.ones((1, N)))))

            # get l:ax + by +c = 0
            l2_a_coes = l2s[0,:]
            l2_b_coes = l2s[1,:]
            l2_c_coes = l2s[2,:]
            #calculate dists
            dists2 = abs(l2_a_coes*pts2[:, 0] + l2_b_coes*pts2[:,1] + l2_c_coes)
            dists2 = dists2/np.sqrt(l2_a_coes**2 + l2_b_coes**2)
            
            inliers_local = dists1+dists2 < threshold
            inliners_cnt = np.sum(inliers_local)
            if max_inliers_cnt < inliners_cnt:
                max_inliers_cnt = inliners_cnt
                #F = F7
                inliers = inliers_local
                #print('dists1=')
                #print(dists1)
                #print('dists2=')
                #print(dists2)
    
    
    print(max_inliers_cnt/N)
        
    
    F = eightpoint(pts1[inliers], pts2[inliers], M)
    
    return F, inliers        
    



'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta != 0:
        n = r / theta
    else:
        n = r
    n_cross = np.array([[      0,-n[2,0], n[1,0]],
                        [ n[2,0],      0,-n[0,0]],
                        [-n[1,0], n[0,0],      0]])

    n_cross_square = np.dot(n, n.T) - np.sum(n**2)*np.identity(3)

    R = np.identity(3) + np.sin(theta)*n_cross + (1.0-np.cos(theta))*n_cross_square

    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    A = (R - R.T)/2
    tho = np.array([[A[2, 1]], [A[0, 2]], [A[1, 0]]])
    s = np.linalg.norm(tho)
    c = (R[0, 0]+R[1, 1]+R[2, 2]-1)/2

    if s == 0 and c == 1:
        return np.zeros((3, 1))

    elif s == 0 and c == -1:
        tmpR = R + np.identity(3)
        v = None
        for i in range(3):
            if np.linalg.norm(tmpR[:,i]) != 0:
                v = tmpR[:, i]
                break
        u = v/np.linalg.norm(v)
        r = u*np.pi
        r = np.reshape(3,1)
        r1 = r[0,0]
        r2 = r[1,0]
        r3 = r[2,0]
        if np.linalg.norm(r) == np.pi and ((r1 == 0 and r2 == 0 and r3 < 0)
                                               or (r1 == 0 and r2 < 0) or (r1 < 0)):
            r = -r
        
        return r
    else:
        u = tho / s
        theta = np.arctan2(np.float(s), np.float(c))
        r = u*theta
        return r



'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    P  = x[:-6]
    r2 = x[-6:-3]
    t2 = x[-3:]

    N = P.shape[0]//3
    P = P.reshape(N, 3)
    
    #r2 t2 : 3x1
    r2 = r2.reshape(3, 1)
    t2 = t2.reshape(3, 1)
    
    #Get R2:3x3
    R2 = rodrigues(r2)
    #M2 = [R2|t2] : 3x4
    M2 = np.hstack((R2, t2))

    #Get C1 & c2: 3x4
    C1 = np.dot(K1, M1)
    C2 = np.dot(K2, M2)

    #Get homogeneous term P:4xN
    P = np.vstack((P.T, np.ones((1, N))))
    
    #p1_hat, p2_hat:Nx2
    p1_hat = np.dot(C1, P)
    p1_hat /= p1_hat[-1, :]
    p1_hat = p1_hat[:2, :].T
    p2_hat = np.dot(C2, P)
    p2_hat /= p2_hat[-1, :]
    p2_hat = p2_hat[:2, :].T

    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])
    residuals = residuals.reshape(residuals.shape[0],1)

    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    
    #M2 = [R2|t2] 3x4
    #R2_init: 3x3
    R2_init = M2_init[:,:3]
    #t2_init: 3x1
    t2_init = M2_init[:,-1]
    #r2_init: 3x1
    r2_init = invRodrigues(R2_init)
    
    #concatenate
    x = np.concatenate([P_init.reshape(-1), r2_init.reshape(-1), t2_init.reshape(-1)])

    #Apply opitimizer
    func = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x)**2).sum()
    res = scipy.optimize.minimize(func, x, method='L-BFGS-B')
    
    #Get optimized_x
    optimized_x = res.x

    P = optimized_x[:-6]
    r2 = optimized_x[-6:-3]
    t2 = optimized_x[-3:]

    N = P.shape[0] // 3
    P2 = P.reshape(N, 3)

    r2 = r2.reshape(3, 1)
    t2 = t2.reshape(3, 1)
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    return M2, P2

