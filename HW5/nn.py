import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None
    bound = np.sqrt(6.0/(in_size+out_size))

    W = np.random.uniform(low=-bound, high=bound,size=(in_size, out_size))
    b = np.zeros(out_size)

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    pre_act = np.dot(X,W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row

def softmax(x):
    res = None

    #Get max_xi from each row (max_xi:Nx1)
    max_xi = np.max(x, axis=1)
    max_xi = np.expand_dims(max_xi, axis=1)

    #Get shift_xs: NxD
    shift_xs = x.copy()
    shift_xs -= max_xi

    #Get exp_terms (exp_terms:NxD)
    exp_terms = np.exp(shift_xs)

    #Get sum_exp_terms_row: Nx1
    sum_exp_terms_row = np.sum(exp_terms, axis=1)
    sum_exp_terms_row = np.expand_dims(sum_exp_terms_row, axis=1)

    #Get res = exp_terms / sum_exp_terms_row (res:NxD)
    res = exp_terms / sum_exp_terms_row

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss = -np.sum(y*np.log(probs))

    acc = 0

    N = y.shape[0]

    for i in range(N):
        predict_idx = np.argmax(probs[i])
        if y[i, predict_idx] == 1:
            acc += 1
    acc /= N

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    grad_W = np.zeros(W.shape)
    grad_b = np.zeros(b.shape)
    grad_X = np.zeros(X.shape)


    delta_mat = delta * activation_deriv(post_act)

    #Go throgh each example
    N = X.shape[0]

    for i in range(N):
        grad_W += np.dot(X[i,:].reshape(X[i,:].shape[0], 1),delta_mat[i,:].reshape(1, delta_mat[i,:].shape[0]))
        grad_b += delta_mat[i,:]
        grad_X[i, :] = np.dot(W, delta_mat[i, :].reshape(delta_mat[i, :].shape[0], 1)).reshape([-1])

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X



############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]

def get_random_batches(x,y,batch_size):
    batches = []
    N = x.shape[0]
    permutation_indice = np.random.permutation(N)
    num_batches = N // batch_size


    for i in range(num_batches):
        selected_indice = permutation_indice[i*batch_size:(i+1)*batch_size]
        batch_x = [x[i] for i in selected_indice]
        batch_x = np.array(batch_x)
        batch_y = [y[i] for i in selected_indice]
        batch_y = np.array(batch_y)
        batches.append((batch_x, batch_y))
    
    return batches

