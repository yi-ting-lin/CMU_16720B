import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
	# put your implementation here
    
	
	
	p = np.zeros(6)
	delta_p = np.array([1,1,1,1,1,1])
	
	#initial values
	epison = 0.001
	H, W = It.shape
	x1, y1, x2, y2 = 0, 0, W-1, H-1

	#get the interpolation operators
	inter_op_It1 	= RectBivariateSpline(np.arange(H), np.arange(W), It1)

	#debug info
	num_iter = 0
	max_num_iter = 200

	while np.dot(delta_p, delta_p.T) >= epison:
		
		#Generate the warping function for I(W(x:P))
		X = np.arange(x1, x2 + epison)
		Y = np.arange(y1, y2 + epison)
		X, Y = np.meshgrid(X, Y)
		
		#Warping to I(W(x:P))
		I_W_x = (1.0 + p[0]) * X + p[1] * Y + p[2]
		I_W_y = p[3] * X + (1.0 + p[4]) * Y + p[5]
		valid_indices = (I_W_x > 0) & (I_W_x < W) & (I_W_y > 0) & (I_W_y < H)
		
		#Get the warped I_W_x, I_W_y in valid range
		I_W_x = I_W_x[valid_indices]
		I_W_y = I_W_y[valid_indices]
		
		#Get the original X, Y in valid range
		X = X[valid_indices].reshape(-1)
		Y = Y[valid_indices].reshape(-1)

		#Generate image term: I(W(x:p))
		I_W_x_p = inter_op_It1.ev(I_W_y, I_W_x)

		#Generate grad_x and grad_y
		grad_x = inter_op_It1.ev(I_W_y, I_W_x, dx=0, dy=1).reshape(-1)
		grad_y = inter_op_It1.ev(I_W_y, I_W_x, dx=1, dy=0).reshape(-1)

		#Calculate A: N*6
		A = []
		A.append(np.multiply(grad_x, X))
		A.append(np.multiply(grad_x, Y))
		A.append(grad_x)
		A.append(np.multiply(grad_y, X))
		A.append(np.multiply(grad_y, Y))
		A.append(grad_y)
		A = np.array(A)
		A = A.T
		

		#Calculate the difference matrix b: N*1
		b = It[valid_indices] - I_W_x_p
		b = b.reshape(-1)
		
		#calculate delta_p: 2*1 and update p = p + delta_p
		delta_p = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T).dot(b)
		p += delta_p
		
		if num_iter >= max_num_iter:
			print('loop over max_num_iter!')
			break

	#update M using p
	M = np.array([[1.0+p[0], p[1], p[2]], [p[3], 1.0+p[4], p[5]]])

	return M