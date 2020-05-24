import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']


max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
input_layer_size = train_x.shape[1]
initialize_weights(input_layer_size, hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'hidden1')
initialize_weights(hidden_size, hidden_size, params, 'hidden2')
initialize_weights(hidden_size, input_layer_size, params, 'output')


#Get extra space params['m_'+k]
keys = [k for k in params.keys()]
for k in keys:
    params['m_' + k] = np.zeros(params[k].shape)

#for ploting
train_losses = []
epochs = [i for i in range(max_iters)]
num_train_data = train_x.shape[0]

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        # forward
        h1      = forward(X=xb, params=params, name='layer1', activation=relu)
        h2      = forward(X=h1, params=params, name='hidden1', activation=relu)
        h3      = forward(X=h2, params=params, name='hidden2', activation=relu)
        x_out   = forward(X=h3, params=params, name='output', activation=sigmoid)

        # loss for training
        # be sure to add loss and accuracy to epoch totals
        
        loss = np.sum((x_out-xb)**2)
        total_loss += loss

        # backward
        d1 = 2*(x_out-xb)
        d2 = backwards(delta=d1, params=params, name='output', activation_deriv=sigmoid_deriv)
        d3 = backwards(delta=d2, params=params, name='hidden2', activation_deriv=relu_deriv)
        d4 = backwards(delta=d3, params=params, name='hidden1', activation_deriv=relu_deriv)
        backwards(delta=d4, params=params, name='layer1', activation_deriv=relu_deriv)

        #update weights
        for k in params.keys():
            if '_' not in k:
                params['m_'+k] = 0.9*params['m_'+k] - learning_rate*params['grad_'+k]
                params[k] += params['m_'+k] 

    #normalize loss by num_train_data
    total_loss /= num_train_data
    train_losses.append(total_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
# ploting for Q5.2
#ploting
import matplotlib.pyplot as plt
plt.figure('Training Loss')
plt.plot(epochs, train_losses, 'r')
    
plt.xlabel('# of epoch')
plt.ylabel('Training Loss')
plt.show()


#Save weights
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q5_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
import pickle
params = pickle.load(open('q5_weights.pickle', 'rb'))
'''
# Q5.3.1
import matplotlib.pyplot as plt
h1              = forward(X=valid_x, params=params, name='layer1', activation=relu)
h2              = forward(X=h1, params=params, name='hidden1', activation=relu)
h3              = forward(X=h2, params=params, name='hidden2', activation=relu)
valid_x_out     = forward(X=h3, params=params, name='output', activation=sigmoid)

num_valid_data = valid_x.shape[0]

ori_crop = None
new_crop = None

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
        new_crop = valid_x_out[i]
        #reshape to 32x32
        new_crop = new_crop.reshape((32,32))
        #Take transport
        new_crop = new_crop.T
        plt.imshow(new_crop)
        plt.show()


# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
avg_PSNR = 0

for i in range(num_valid_data):
    avg_PSNR += psnr(valid_x[i], valid_x_out[i])

avg_PSNR /= num_valid_data

print('avg_PSNR: ' + str(avg_PSNR))
