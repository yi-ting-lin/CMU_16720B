'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import submission
import helper
import findM2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


#load correspondence
corresp = np.load('../data/some_corresp.npz')
corresp = dict(corresp)

pts1 = corresp['pts1']
pts2 = corresp['pts2']


#load images
img1 = plt.imread('../data/im1.png')
img2 = plt.imread('../data/im2.png')

#Get M
M = max(img1.shape[0],img1.shape[1],img2.shape[0],img2.shape[1])

#Estimate F
F = submission.eightpoint(pts1=pts1, pts2=pts2, M=M)

#load intrinsics
intrinsics = np.load('../data/intrinsics.npz')
intrinsics = dict(intrinsics)
K1 = intrinsics['K1']
K2 = intrinsics['K2']
#Estimate E
E = submission.essentialMatrix(F, K1, K2)


#load hand-pice correspondence x1 y1
corresp = np.load('../data/templeCoords.npz')
corresp = dict(corresp)
x1s = corresp['x1']
y1s = corresp['y1']


#generate x2 y2 using epipolarCorrespondence from x1 y1
num_corresp = x1s.shape[0]
x2s = []
y2s = []
for x1, y1 in zip(x1s, y1s):
    x2, y2 = submission.epipolarCorrespondence(img1, img2, F, x1[0], y1[0])
    x2s.append(x2)
    y2s.append(y2)

x2s = np.array(x2s).reshape((num_corresp, 1))
y2s = np.array(y2s).reshape((num_corresp, 1))

#Get M1 & C1
M1 = np.array([[1,0,0,0],
			   [0,1,0,0],
			   [0,0,1,0]])
C1 = np.dot(K1, M1)	

#Get M2 & C2 & P from test_M2_solution
M2, C2, P = findM2.test_M2_solution(np.hstack((x1s,y1s)), np.hstack((x2s,y2s)), intrinsics, M)

#Save results into q4_2.npz
np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)

# plot the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#set min & max
#upper_bound = np.max(P)
#lower_bound = np.min(P)

ax.set_xlim3d(np.amin(P[:, 0]), np.amax(P[:, 0]))
ax.set_ylim3d(np.amin(P[:, 1]), np.amax(P[:, 1]))
ax.set_zlim3d(np.amin(P[:, 2]), np.amax(P[:, 2]))


ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='r', marker='.')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()