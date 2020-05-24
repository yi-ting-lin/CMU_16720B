import numpy as np
import skimage
import multiprocessing
import threading
import queue
import os,time
import math
import visual_words
import scipy

def worker(args):
    file_path, i , dictionary, layer_num, K, is_train = args
    print("[Feature Extraction] Image #%d is under processing" %i)
    feature = get_image_feature(file_path, dictionary, layer_num, K)

    if is_train == True:
        np.save('train_features' + str(i) + '.npy', feature)
    else:
        np.save('test_features' + str(i) + '.npy', feature)
    
    print("[Feature Extraction] Image #%d is finished" %i)

def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K, 3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''
    train_data = dict(np.load("../data/train_data.npz"))
    dictionary = np.load("dictionary.npy")
    

    #img_path_seq = train_data['files'][:30]
    img_path_seq = train_data['files']
    #Constants
    N = len(img_path_seq)
    K = len(dictionary)
    M = 21*K #L=2 for SPM
    SPM_layer_num = 3

    #labels
    labels = train_data['labels'][:]

    print('There are ' + str(N) + 'train data to be processed')
    features = np.zeros((N, M))
    pool = multiprocessing.Pool(num_workers)
    pool.map(worker, [('../data/'+img_path_seq[i], i , dictionary, SPM_layer_num, K, True) for i in range(N)])

    for i in range(N):
        features[i] = np.load('train_features' + str(i) + '.npy')

    np.savez('trained_system.npz',features=features, labels=labels, dictionary=dictionary, SPM_layer_num=SPM_layer_num)

def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = dict(np.load("../data/test_data.npz"))
    trained_system = dict(np.load("trained_system.npz"))

    #loading training data
    train_features  = trained_system['features']
    train_labels    = trained_system['labels']
    dictionary      = trained_system['dictionary']
    SPM_layer_num   = trained_system['SPM_layer_num']


    #Testing parameters
    img_path_seq    = test_data['files'][:]
    test_labels     = test_data['labels'][:]
    N = len(img_path_seq)
    K = len(dictionary)
    M = 21*K #L=2 for SPM

    #Feature extraction for testing
    print('There are ' + str(N) + 'test data to be processed')
    pool = multiprocessing.Pool(num_workers)
    pool.map(worker, [('../data/'+img_path_seq[i], i , dictionary, SPM_layer_num, K, False) for i in range(N)])


    correct = 0
    conf = np.zeros((8,8))
    for i in range(N):
        test_feature = np.load('test_features' + str(i) + '.npy')
        sim = distance_to_set(test_feature, train_features)
        predict_label = train_labels[np.argmax(sim)]
        conf[int(predict_label)][int(test_labels[i])] += 1
        if predict_label == test_labels[i]:
            correct += 1
    accuracy = correct / N

    print(accuracy)
    print(conf)

    return conf, accuracy

def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''


    #image parsing
    image = skimage.io.imread(file_path)
    image[:,:,:] = image[:,:,:3]
    image = image.astype('float')/255
    labImage = skimage.color.rgb2lab(image)

    #get word_map
    word_map = visual_words.get_visual_words(labImage, dictionary)
    
    #get features
    hists = get_feature_from_wordmap_SPM(word_map, 3, K)

    return hists


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N, K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    N = len(histograms)
    K = len(word_hist)

    sim = scipy.spatial.distance.cdist(histograms, word_hist.reshape(1, K), lambda u, v: (np.minimum(u, v)).sum())
    #K = len(word_hist)
    #sim = np.zeros(N)
    #for i in range(N):
    #    sim[i] += (np.minimum(histograms[i], word_hist)).sum()

    return sim


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    hist = np.zeros(dict_size)

    for x in range(wordmap.shape[0]):
        for y in range(wordmap.shape[1]):
            hist[int(wordmap[x, y])] += 1
    
    hist /= hist.sum()
    return hist
   

def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3) = 21K (1+4+16 when L = 2)
    '''

    #Fine (4x4)
    H = wordmap.shape[0]
    W = wordmap.shape[1]
    divid_H = int(H/4)
    divid_W = int(W/4)
    hists = np.zeros(dict_size*21)
    for i in range(16):
        start_row = int((i % 4) * divid_H)
        start_col = int((int)(i / 4) * divid_W)
        hists[i*dict_size:(i+1)*dict_size] = get_feature_from_wordmap(wordmap[start_row:start_row+divid_H, start_col:start_col+divid_W], dict_size)


    #medium (2x2)
    hists[16*dict_size:(17)*dict_size] = hists[0*dict_size:(0+1)*dict_size] + hists[1*dict_size:(1+1)*dict_size] + hists[4*dict_size:(4+1)*dict_size] + hists[5*dict_size:(5+1)*dict_size]
    hists[17*dict_size:(18)*dict_size] = hists[2*dict_size:(2+1)*dict_size] + hists[3*dict_size:(3+1)*dict_size] + hists[6*dict_size:(6+1)*dict_size] + hists[7*dict_size:(7+1)*dict_size]
    hists[18*dict_size:(19)*dict_size] = hists[8*dict_size:(8+1)*dict_size] + hists[9*dict_size:(9+1)*dict_size] + hists[12*dict_size:(12+1)*dict_size] + hists[13*dict_size:(13+1)*dict_size]
    hists[19*dict_size:(20)*dict_size] = hists[10*dict_size:(10+1)*dict_size] + hists[11*dict_size:(11+1)*dict_size] + hists[14*dict_size:(14+1)*dict_size] + hists[15*dict_size:(15+1)*dict_size]


    #courase (1x1)
    hists[20*dict_size:(21)*dict_size] = hists[16*dict_size:(17)*dict_size] + hists[17*dict_size:(18)*dict_size] + hists[18*dict_size:(19)*dict_size] + hists[19*dict_size:(20)*dict_size]
    #print(hists)
    #print(hists.sum())
    #normalization
    hists[0:16*dict_size] /= 2
    hists[16*dict_size:20*dict_size] /= 4
    hists[20*dict_size:] /= 4

    hists/= hists.sum()

    #distance_to_set(hists, hists.reshape(1,len(hists)))

    return hists