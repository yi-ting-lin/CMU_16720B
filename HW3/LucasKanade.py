import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    # Put your implementation here
    
	#initial values
	p = p0
	epison = 0.005
	small_val_for_inter = 0.0001
	delta_p = np.array([1,1])
	x1, y1, x2, y2 = rect

	H, W = It.shape

	#get the interpolation operators
	inter_op_It 	= RectBivariateSpline(np.arange(H), np.arange(W), It)
	inter_op_It1 	= RectBivariateSpline(np.arange(H), np.arange(W), It1)


	#debug info
	num_iter = 0
	max_num_iter = 2000


	while np.dot(delta_p, delta_p.T) >= epison:
		#Generate template term: T(x)
		T_x = np.arange(x1, x2 + small_val_for_inter)
		T_y = np.arange(y1, y2 + small_val_for_inter)
		T_x, T_y = np.meshgrid(T_x, T_y)
		Tx = inter_op_It.ev(T_y, T_x)

		#Generate image term: I(W(x:p))
		I_W_x = np.arange(x1+p[0], x2+small_val_for_inter+p[0])
		I_W_y = np.arange(y1+p[1], y2+small_val_for_inter+p[1])
		I_W_x, I_W_y = np.meshgrid(I_W_x, I_W_y)
		I_W_x_p = inter_op_It1.ev(I_W_y, I_W_x)

		
		#Calculate the difference matrix d = T(x) - I(W(x:p)): N*1
		d = Tx - I_W_x_p
		d = d.reshape(-1)

		#Generate grad_x and grad_y and flatten into grad_I: N*2
		grad_x = inter_op_It1.ev(I_W_y, I_W_x, dx=0, dy=1).reshape(-1)
		grad_y = inter_op_It1.ev(I_W_y, I_W_x, dx=1, dy=0).reshape(-1)
		grad_I = []
		grad_I.append(grad_x)
		grad_I.append(grad_y)
		grad_I = np.array(grad_I).T
		
		#calculate delta_p: 2*1 and update p = p + delta_p
		delta_p = np.dot(np.linalg.inv(np.dot(grad_I.T, grad_I)), grad_I.T).dot(d)
		p += delta_p
		
		if num_iter >= max_num_iter:
			print('loop over max_num_iter!')
			break

	return p

