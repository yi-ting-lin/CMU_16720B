import numpy as np
import matplotlib.pyplot as plt
import submission
import helper

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def test_M2_solution(pts1, pts2, intrinsics, M):
	'''
	Estimate all possible M2 and return the correct M2 and 3D points P
	:param pred_pts1:
	:param pred_pts2:
	:param intrinsics:
	:param M: a scalar parameter computed as max (imwidth, imheight)
	:return: M2, the extrinsics of camera 2
			 C2, the 3x4 camera matrix
			 P, 3D points after triangulation (Nx3)
	'''
	

	#Get K1 K2
	intrinsics = dict(intrinsics)
	K1 = intrinsics['K1']
	K2 = intrinsics['K2']

	#Get M1
	M1 = np.array([[1,0,0,0],
				   [0,1,0,0],
				   [0,0,1,0]])
	C1 = np.dot(K1, M1)	

	#Get F using eight point algorithm
	F = submission.eightpoint(pts1=pts1, pts2=pts2, M=M)

	#Get E
	E = submission.essentialMatrix(F, K1, K2)

	#Get M2s : 4 possible M2
	M2s = helper.camera2(E)

	M2 = None
	P = None
	C2 = None

	for i in range(M2s.shape[2]):
		M2_local = M2s[:,:,i]
		C2_local = np.dot(K2, M2_local)
		P_local, err = submission.triangulate(C1, pts1, C2_local, pts2)
		print(err)
		#get minimal value of depth
		min_z = np.amin(P_local[:,2])
		if min_z > 0:
			M2 = M2_local
			P = P_local
			C2 = C2_local
			break
	
	return M2, C2, P


if __name__ == '__main__':
	data = np.load('../data/some_corresp.npz')
	pts1 = data['pts1']
	pts2 = data['pts2']
	intrinsics = np.load('../data/intrinsics.npz')
	img1 = plt.imread('../data/im1.png')
	img2 = plt.imread('../data/im2.png')

	M = max(img1.shape[0],img1.shape[1],img2.shape[0],img2.shape[1])


	M2, C2, P = test_M2_solution(pts1, pts2, intrinsics,M)
	np.savez('q3_3.npz', M2=M2, C2=C2, P=P)
