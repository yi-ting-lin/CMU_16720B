import numpy as np
import multiprocessing
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import util
import random
import imageio


def worker_filter_response(args):
    alpha, img_path_seq, i = args
    print("Image #%d is under processing" %i)
    single_filter_response = compute_dictionary_one_image(alpha, img_path_seq, i)
    np.save('single_filter_response' + str(i) + '.npy', single_filter_response)
   
    print("Image #%d is finished" %i)


def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''

    scalars = [1, 2, 4, 8, 8*1.414]
    filter_respnses = np.ndarray((image.shape[0], image.shape[1], image.shape[2]*20))
    
    for i in range(len(scalars)):
        img_g     = scipy.ndimage.gaussian_filter(image, scalars[i])
        img_loG   = scipy.ndimage.gaussian_laplace(image, scalars[i])
        img_gx    = scipy.ndimage.filters.gaussian_filter(image,scalars[i],(0,1,0))
        img_gy    = scipy.ndimage.filters.gaussian_filter(image, scalars[i], (1,0,0))
        for x in range(img_g.shape[0]):
            for y in range(img_g.shape[1]):
                for z in range(3):
                    filter_respnses[x, y, 3*i     +z] = img_g[x, y, z]
                    filter_respnses[x, y, 3*(i+5) +z] = img_loG[x, y, z]
                    filter_respnses[x, y, 3*(i+10)+z] = img_gx[x, y, z]
                    filter_respnses[x, y, 3*(i+15)+z] = img_gy[x, y, z]


    return filter_respnses
def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    '''
    filter_responses = extract_filter_responses(image)

    word_map = np.ndarray((image.shape[0], image.shape[1]))
    for x in range(filter_responses.shape[0]):
        for y in range(filter_responses.shape[1]):
            dists = scipy.spatial.distance.cdist(dictionary, filter_responses[x][y].reshape(1,60))
            word_map[x, y] = np.argmin(dists)
    
    return word_map



def compute_dictionary_one_image(alpha, img_path_seq, i):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''

    

  
    image = imageio.imread('../data/' + img_path_seq[i])

    if image.shape[2] != 3:
        image[:,:,:] = image[:,:,:3]

    image = image.astype('float')/255
    labImage = skimage.color.rgb2lab(image)
    filter_responses = extract_filter_responses(labImage)
    #filter_responses: numpy.ndarray of shape (H, W, 3F)
     

    sampled_response = np.ndarray((alpha, 3*20))


    #sample_x = np.random.permutation(alpha)
    #sample_y = np.random.permutation(alpha)

    sample_x = np.random.permutation(filter_responses.shape[0])
    sample_y = np.random.permutation(filter_responses.shape[1])

    for idx in range(alpha):
        #print(str(sample_x[idx]) + ', ' + str(sample_y[idx]))
        sampled_response[idx][:] = filter_responses[sample_x[idx]][sample_y[idx]][:]

    return sampled_response


def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''

    train_data = dict(np.load("../data/train_data.npz"))
    img_path_seq = train_data['files']

  
    alpha = 120
    K=150
    num_train_data = len(img_path_seq)
    filter_responses = np.ndarray((alpha * num_train_data, 3*20))
    
    
    print("There are %d images to be processed..." %num_train_data)

    
    
    pool = multiprocessing.Pool(num_workers)
    pool.map(worker_filter_response, [(alpha, img_path_seq, i) for i in range(num_train_data)])

    #print("Generated filter responses shape = " + str(filter_responses.shape))
    

    '''
    for i in range(num_train_data):
        filter_response = np.load('single_filter_response' + str(i) + '.npy')
        for x in range(filter_response.shape[0]):
            for y in range(filter_response.shape[1]):
                filter_responses[i*alpha+x][y] = filter_response[x][y]
    
    np.save('filter_responses.npy', filter_responses)
    '''


    for i in range(num_train_data):
        filter_responses[i*alpha:(i+1)*alpha] = np.load('single_filter_response' + str(i) + '.npy')

    np.save('filter_responses.npy', filter_responses)
    print("Generated filter responses shape = " + str(filter_responses.shape))

    kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=num_workers).fit(filter_responses)
    dictionary =  kmeans.cluster_centers_
    np.save('dictionary.npy', dictionary)




