import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
	M 		= np.identity(3)
	delta_p = np.array([1,1,1,1,1,1])
	
	#initial values
	epison = 0.001
	H, W = It.shape
	x1, y1, x2, y2 = 0, 0, W-1, H-1

	#get the interpolation operators
	inter_op_It 	= RectBivariateSpline(np.arange(H), np.arange(W), It)
	inter_op_It1 	= RectBivariateSpline(np.arange(H), np.arange(W), It1)

	#debug info
	num_iter = 0
	max_num_iter = 200


	#Generate the original coordinates X & Y
	X = np.arange(x1, x2 + epison)
	Y = np.arange(y1, y2 + epison)
	X, Y = np.meshgrid(X, Y)

	#Generate grad_x and grad_y of template
	grad_x = inter_op_It.ev(Y, X, dx=0, dy=1)
	grad_y = inter_op_It.ev(Y, X, dx=1, dy=0)
	grad_x = grad_x.reshape(-1)
	grad_y = grad_y.reshape(-1)

	#Calculate A: N*6 still need to pick the valid region
	A = []
	A.append(np.multiply(grad_x, X.reshape(-1)))
	A.append(np.multiply(grad_x, Y.reshape(-1)))
	A.append(grad_x)
	A.append(np.multiply(grad_y, X.reshape(-1)))
	A.append(np.multiply(grad_y, Y.reshape(-1)))
	A.append(grad_y)
	A = np.array(A)
	A = A.T

	#print(A.shape)

	while np.dot(delta_p, delta_p.T) >= epison:
		
		#Warping to I(W(x:P))
		I_W_x = M[0,0] * X + M[0,1] * Y + M[0,2]
		I_W_y = M[1,0] * X + M[1,1] * Y + M[1,2]
		valid_indices = (I_W_x > 0) & (I_W_x < W) & (I_W_y > 0) & (I_W_y < H)
		
		#Get the warped I_W_x, I_W_y in valid range
		I_W_x = I_W_x[valid_indices]
		I_W_y = I_W_y[valid_indices]
		
		#Generate image term: I(W(x:p))
		I_W_x_p = inter_op_It1.ev(I_W_y, I_W_x)
		
		#Get the valid_indices of A
		valid_A = A[valid_indices.reshape(-1)]

		#Calculate the difference matrix b: N*1
		b =  I_W_x_p - It[valid_indices]
		b = b.reshape(-1)
		
		#calculate delta_p: 2*1
		delta_p = np.dot(np.linalg.inv(np.dot(valid_A.T, valid_A)), valid_A.T).dot(b)

		#update transformation Matrix M = [1+p0, p1, p2]  using delta_M =   [1+detal_p0, delta_p1, delta_p2] 
		#								  [p3 ,1+p4, p5]					[detal_p3, 1+delta_p4, delta_p5] 
		#								  [0,     0,  1] 					[0       ,          0,        1] 
		delta_M = np.identity(3)
		delta_M[0,0] += delta_p[0]
		delta_M[0,1] += delta_p[1]
		delta_M[0,2] += delta_p[2]
		delta_M[1,0] += delta_p[3]
		delta_M[1,1] += delta_p[4]
		delta_M[1,2] += delta_p[5]
		
		M = np.dot(M, np.linalg.inv(delta_M))

		if num_iter >= max_num_iter:
			print('loop over max_num_iter!')
			break

	return M[:2,:]