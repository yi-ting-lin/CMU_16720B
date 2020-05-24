import numpy as np
import scipy.ndimage
import os

def extract_deep_feature(x, vgg16_weights):

	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (3, H, W)
	* vgg16_weights: list of shape (L, 3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''

	feat = x
	l = vgg16_weights
	seq_layers = 31
	#seq_layers = 1
	classify_layers = 3
	
	for i in range(seq_layers):
		if l[i][0] == 'conv2d':
			weight = l[i][1]
			bias = l[i][2]
			feat = multichannel_conv2d(feat, weight, bias)
		
		if l[i][0] == 'relu':
			feat = relu(feat)
		
		if l[i][0] == 'maxpool2d':
			size = l[i][1]
			feat = max_pool2d(feat, size)
		
		#('sequential operation#' + str(i) + ': ' + l[i][0])
		#print(feat.shape)
	
	#flatten
	
	feat = feat.reshape((25088))
	#print(feat.shape)

	for i in range(seq_layers, seq_layers + classify_layers):
		if l[i][0] == 'linear':
			weight = l[i][1]
			bias = l[i][2]
			feat = linear(feat, weight, bias)

		if l[i][0] == 'relu':
			feat = relu(feat)
		#print('classification operation#' + str(i) + ': ' + l[i][0])
		#print(feat.shape)
	

	return feat
	
	


def multichannel_conv2d(x, weight, bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (input_dim, H, W), where input_dim = n_channels
	* weight: numpy.ndarray of shape (output_dim, input_dim, kernel_size, kernel_size)
	* bias: numpy.ndarray of shape (output_dim), where output_dim = n_filters

	[output]
	* feat: numpy.ndarray of shape (output_dim, H, W)
	'''
	n_channels, H, W = x.shape 	
	n_filters	= weight.shape[0]
	

	feat = np.zeros((n_filters, H, W))
	
	
	
	for i in range(n_filters):
		for channel in range(n_channels):
			filp_w = np.flip(weight[i][channel])
			feat[i] += scipy.ndimage.convolve(x[channel], filp_w, mode='constant', cval=0.0)
		feat[i] += bias[i]
	

	#print('x.shape:' + str(x.shape))
	#print('weight.shape:' + str(weight.shape))
	#print('bias.shape:' + str(bias.shape))
	
	return feat

def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	x[x<0]=0
	return x
def max_pool1d(x, size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H, W)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size, W/size)
	'''

	H, W = x.shape

	sub_H = H // size
	sub_W = W // size
	y = x[:sub_H*size, :sub_W*size].reshape(sub_H, size, sub_W, size).max(axis=(1, 3))
	return y


def max_pool2d(x, size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (input_dim, H, W)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (input_dim, H/size, W/size)
	'''
	input_dim, H, W  = x.shape
	sub_H = H // size
	sub_W = W // size
	y = np.zeros((input_dim, sub_H, sub_W))
	
	for i in range(input_dim):
		y[i] = max_pool1d(x[i], size)
		
	return y
	

def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	W_x = W.dot(x)
	y = W_x + b
	
	return y
