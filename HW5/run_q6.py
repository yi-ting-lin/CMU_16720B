import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

#train_x: n x 1024

dim = 32
# do PCA
#substract the mean of data
num_train_data = train_x.shape[0]
train_data_means = (np.sum(train_x,axis=0) / num_train_data)
train_x_zero_offst = train_x.copy()
train_x_zero_offst -= train_data_means

#apply svd on train_x_zero_offst
u, s, vt = np.linalg.svd(train_x_zero_offst)


#use the prinipal component (proj_mat: 32x1024)
proj_mat = vt[:dim, :]

# rebuild a low-rank version # 1024x32
lrank = np.dot(train_x_zero_offst, proj_mat.T)

# rebuild it recon:1024x1024
recon = np.dot(lrank, proj_mat)
recon += train_data_means

# build valid dataset
#substract the mean of data
num_valid_data = valid_x.shape[0]
valid_data_means = (np.sum(valid_x,axis=0) / num_valid_data)
valid_x_zero_offst = valid_x.copy()
valid_x_zero_offst -= valid_data_means

# rebuild a low-rank version # 1024x32
lrank_valid = np.dot(valid_x_zero_offst, proj_mat.T)

# recon_valid = None
# rebuild it recon:1024x1024
recon_valid = np.dot(lrank_valid, proj_mat)
recon_valid += valid_data_means


# visualize the comparison and compute PSNR
for i in range(num_valid_data):
    if i%700 == 0 or i%700 == 1:
        #Original crop
        plt.subplot(2,2,1)
        #Get the original crop
        ori_crop = valid_x[i]
        #reshape to 32x32
        ori_crop = ori_crop.reshape((32,32))
        #Take transport
        ori_crop = ori_crop.T
        plt.imshow(ori_crop)

        #Trained crop
        plt.subplot(2,2,2)
        #Get the original crop
        new_crop = recon_valid[i]
        #reshape to 32x32
        new_crop = new_crop.reshape((32,32))
        #Take transport
        new_crop = new_crop.T
        plt.imshow(new_crop)
        plt.show()

from skimage.measure import compare_psnr as psnr
# evaluate PSNR
avg_PSNR = 0

for i in range(num_valid_data):
    avg_PSNR += psnr(valid_x[i], recon_valid[i])

avg_PSNR /= num_valid_data

print('avg_PSNR: ' + str(avg_PSNR))
