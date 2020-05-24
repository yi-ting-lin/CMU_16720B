import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
import scipy.ndimage
from scipy.interpolate import RectBivariateSpline
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
	#Threshold
	diff_threshold = 0.1
	
	#Get matrix M and add [0,0,1] to it's last row
	M = LucasKanadeAffine(image1, image2)
	#M = InverseCompositionAffine(image1, image2)
	M = np.vstack((M, np.array([0,0,1])))

	#Calculate the invalid region using inv_M and assign these region = 0
	H, W = image1.shape
	x1, y1, x2, y2 = 0, 0, W-1, H-1
	X = np.arange(x1, x2 + diff_threshold)
	Y = np.arange(y1, y2 + diff_threshold)
	X, Y = np.meshgrid(X, Y)


	#Get interpolation operator
	inter_op_image1 	= RectBivariateSpline(np.arange(H), np.arange(W), image1)

	#Get inverse M and try to warp image1 to image2 using inv_M
	inv_M = np.linalg.inv(M)
	warp_X = inv_M[0,0] * X + inv_M[0,1] * Y + inv_M[0,2]
	warp_Y = inv_M[1,0] * X + inv_M[1,1] * Y + inv_M[1,2]
	invalid_region = (warp_X < 0) | (warp_X > x2) | (warp_Y < 0) | (warp_Y > y2)
	warp_image1 = inter_op_image1.ev(warp_Y, warp_X)
	warp_image2 = image2.copy()
	warp_image1[invalid_region] = 0
	warp_image2[invalid_region] = 0


	#calculate the difference between warp_image1 and image2
	mask = np.ones(image1.shape, dtype=bool)
	diff = abs(warp_image1 - warp_image2)
	true_region = diff > diff_threshold	
	mask = np.bitwise_and(mask, true_region)

	#Do erosion & dilation
	kernel1 = np.ones((2,2))
	kernel2 = np.ones((5,5))
	mask = scipy.ndimage.binary_erosion(mask, structure=kernel1).astype(bool)
	mask = scipy.ndimage.binary_dilation(mask, structure=kernel2).astype(bool)

	return mask
