import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']




max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.01
hidden_size = 64

#Get batches
batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,train_y.shape[1],params,'output')

#copy the initialized params
import copy
params_init = copy.deepcopy(params)

#List for ploting
train_accs = []
train_losses = []
valid_accs = []
valid_losses = []
epochs = [i for i in range(max_iters)]



num_train_data = train_x.shape[0]
num_valid_data = valid_x.shape[0]
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    #Run validation data here
    h1 = forward(X=valid_x, params=params, name='layer1', activation=sigmoid)
    probs = forward(X=h1, params=params, name='output', activation=softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)

    valid_loss /= num_valid_data
    valid_accs.append(valid_acc)
    valid_losses.append(valid_loss)


    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # forward
        h1 = forward(X=xb, params=params, name='layer1', activation=sigmoid)
        probs = forward(X=h1, params=params, name='output', activation=softmax)

        # loss for training
        # be sure to add loss and accuracy to epoch totals
        train_loss, train_acc = compute_loss_and_acc(yb, probs)
        total_loss += train_loss
        total_acc += train_acc

        # backward
        d1 = probs - yb
        d2 = backwards(delta=d1, params=params, name='output', activation_deriv=linear_deriv)
        backwards(delta=d2, params=params, name='layer1', activation_deriv=sigmoid_deriv)

        # apply gradient
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']

    total_acc /= batch_num
    #total_loss /= batch_size
    total_loss /= num_train_data
    train_accs.append(total_acc)
    train_losses.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

#ploting
plt.figure('accuracy')
plt.plot(epochs, train_accs, 'r',label='Train data')
plt.plot(epochs, valid_accs, 'b', label='Validation data')
    
plt.xlabel('# of epoch')
plt.ylabel('Averge accuracy')
plt.legend(loc='lower right')
plt.show()

plt.figure('loss')
plt.plot(epochs, train_losses, 'r',label='Train data')
plt.plot(epochs, valid_losses, 'b', label='Validation data')
    
plt.xlabel('# of epoch')
plt.ylabel('Averge cross entropy loss')
plt.legend(loc='upper right')
plt.show()



# run on validation set and report accuracy! should be above 75%
# forward
h1 = forward(X=valid_x, params=params, name='layer1', activation=sigmoid)
valid_probs = forward(X=h1, params=params, name='output', activation=softmax)

# loss
# be sure to add loss and accuracy to epoch totals
_, valid_acc = compute_loss_and_acc(valid_y, valid_probs)

print('Validation accuracy: ',valid_acc)

# run on test set
# forward
h1 = forward(X=test_x, params=params, name='layer1', activation=sigmoid)
test_probs = forward(X=h1, params=params, name='output', activation=softmax)

# loss
# be sure to add loss and accuracy to epoch totals
_, test_acc = compute_loss_and_acc(test_y, test_probs)

print('Testing accuracy: ',test_acc)


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

params = pickle.load(open('q3_weights.pickle','rb'))

fig = plt.figure('Trained Weights')
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 8x8 grid of axes
                 axes_pad=0.1,         # pad between axes in inch.
                 )


for i in range(hidden_size):
    grid[i].imshow(params['Wlayer1'][:, i].reshape(32, 32))  # The AxesGrid object work as a list of axes.
    plt.axis('off')

plt.show()

fig = plt.figure('Initial Weights')
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,         # pad between axes in inch.
                 )


for i in range(hidden_size):
    grid[i].imshow(params_init['Wlayer1'][:, i].reshape(32, 32))  # The AxesGrid object work as a list of axes.
    plt.axis('off')

plt.show()

# Q3.1.4
confusion_matrix_valid = np.zeros((valid_y.shape[1],valid_y.shape[1]))

#Confusion_matrix for validation
for valid_prob, label in zip(valid_probs, valid_y):
    predict_idx = np.argmax(valid_prob)
    actual_idx = np.argmax(label)
    confusion_matrix_valid[predict_idx, actual_idx] += 1


import string
plt.figure('confusion_matrx for validation dataset')
plt.imshow(confusion_matrix_valid,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()

confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

#Confusion_matrix for testing
for test_prob, label in zip(test_probs, test_y):
    predict_idx = np.argmax(test_prob)
    actual_idx = np.argmax(label)
    confusion_matrix[predict_idx, actual_idx] += 1


import string
plt.figure('confusion_matrx for testing dataset')
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()