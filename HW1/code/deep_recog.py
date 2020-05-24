import numpy as np
import multiprocessing
import threading
import queue
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import scipy

#Config
USE_VGG16 = True

def worker(args):
    file_path, i, vgg16_weights, is_train = args
    print("[Deep Feature Extraction] Image #%d is under processing" %i)

	
    image = skimage.io.imread('../data/' + file_path)
    image[:,:,:] = image[:,:,:3]
    image = preprocess_image(image)
    feature = network_layers.extract_deep_feature(image, vgg16_weights)

    if is_train == True:
        np.save('deep_train_features' + str(i) + '.npy', feature)
    else:
        np.save('deep_test_features' + str(i) + '.npy', feature)
    
    print("[Deep Feature Extraction] Image #%d is finished" %i)

def worker_vgg16(args):
    file_path, i, vgg16, is_train = args
    print("[vgg16 Feature Extraction] Image #%d is under processing" %i)
	
    image = skimage.io.imread('../data/' + file_path)
    image[:,:,:] = image[:,:,:3]
    image = preprocess_image(image)
    feature = get_image_feature(image, vgg16)

    if is_train == True:
        np.save('vgg16_train_features' + str(i) + '.npy', feature)
    else:
        np.save('vgg16_test_features' + str(i) + '.npy', feature)
    
    print("[vgg16 Feature Extraction] Image #%d is finished" %i)


def test(vgg16):

    train_data = dict(np.load("../data/train_data.npz"))
    files = train_data['files'][:]
    
    image = skimage.io.imread('../data/' + files[0])
    image[:,:,:] = image[:,:,:3]
    image = preprocess_image(image)
    vgg16_weights = util.get_VGG16_weights()

    feature = network_layers.extract_deep_feature(image, vgg16_weights)

    image = torch.from_numpy(image)
    image = image.double()
    vgg_feature = vgg16.features(image.unsqueeze(0))
    vgg_feature = vgg_feature.flatten()
    vgg_feature = vgg16.classifier[:-3](vgg_feature)

    close = np.isclose(feature, vgg_feature.detach().numpy())
    for element in close:
        if element == False:
             print(element)

def build_recognition_system(vgg16, num_workers=2):

    
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, K)
    * labels: numpy.ndarray of shape (N)
    '''

    
    if USE_VGG16:
        #vgg16.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
        #                               torch.nn.ReLU(),
        #                               torch.nn.Linear(4096, 4096))
        #vgg16.double()
        print(vgg16)

    train_data = dict(np.load("../data/train_data.npz"))
    files = train_data['files'][:100]
    labels = train_data['labels'][:100]

    VGG16_weights = util.get_VGG16_weights()
    N = len(files)
    K = 4096 #VGG output

    print('There are ' + str(N) + ' train data to be processed')
    features = np.zeros((N, K))
    pool = multiprocessing.Pool(num_workers)
    if USE_VGG16:
        pool.map(worker_vgg16, [('../data/'+files[i], i , vgg16, True) for i in range(N)])
    else:
    	pool.map(worker, [('../data/'+files[i], i , VGG16_weights, True) for i in range(N)])
   

	
    for i in range(N):
        if USE_VGG16:
	        features[i] = np.load('vgg16_train_features' + str(i) + '.npy')
        else:
            features[i] = np.load('deep_train_features' + str(i) + '.npy')
		
    np.savez('trained_system_deep.npz', features=features, labels=labels)
	
    

def evaluate_recognition_system(vgg16, num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = dict(np.load("../data/test_data.npz"))
    trained_system = dict(np.load("trained_system_deep.npz"))

    #loading training data
    train_features  = trained_system['features']
    train_labels    = trained_system['labels']

    #Testing parameters
    files    		= test_data['files'][:]
    test_labels     = test_data['labels'][:]
    VGG16_weights 	= util.get_VGG16_weights()
    N = len(files)
    K = 4096 #VGG output

    
    print('There are ' + str(N) + ' test data to be processed')
    pool = multiprocessing.Pool(num_workers)
    if USE_VGG16:
        pool.map(worker_vgg16, [('../data/'+files[i], i , vgg16, False) for i in range(N)])
    else:
    	pool.map(worker, [('../data/'+files[i], i , VGG16_weights, False) for i in range(N)])
    


    correct = 0
    conf = np.zeros((8,8))
    for i in range(N):
        if USE_VGG16:
            test_feature = np.load('vgg16_test_features' + str(i) + '.npy')
        else:
            test_feature = np.load('deep_test_features' + str(i) + '.npy')
        
        sim = distance_to_set(test_feature, train_features)
        predict_label = train_labels[np.argmin(sim)]
        conf[int(predict_label)][int(test_labels[i])] += 1
        if predict_label == test_labels[i]:
            correct += 1
    accuracy = correct / N

    print(accuracy)
    print(conf)

    return conf, accuracy


def preprocess_image(image):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H, W, 3)

    [output]
    * image_processed: torch.Tensor of shape (3, H, W)
    '''
    

    image = image.astype('float')
    image = skimage.transform.resize(image, (224,224,3))
    image = image.astype('float')/255
    mean 	= [0.485, 0.456, 0.406]
    std 	= [0.229, 0.224, 0.225]
    image 	-= mean    #(H,W,3)
    image   /= std
    
    image 	= np.transpose(image, (2,0,1)) #(3,H,W)
	
	
    return image


def get_image_feature(image, vgg16):
#def get_image_feature(image, vgg16):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.
	
    [output]
    * feat: evaluated deep feature
    '''

    #image, vgg16 = args
    image = torch.from_numpy(image)
    image = image.double()
	
    #for parma in vgg16.parameters():
    #parma.requires_grad=False

    #vgg16.features = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    #vgg16.avgpool = torch.nn.Sequential()
    #vgg16.classifier = torch.nn.Sequential()
    #print(vgg16)

 
    #vgg16.double()
    feat = vgg16.features(image.unsqueeze(0))
    #feat = vgg16.avgpool(feat)
    feat = feat.flatten()
    feat = vgg16.classifier[:-3](feat)
    #print(feat.shape)

    return feat.detach().numpy()


def distance_to_set(feature, train_features):
    '''
    Compute distance between a deep feature with all training image deep features.

    [input]
    * feature: numpy.ndarray of shape (K)
    * train_features: numpy.ndarray of shape (N, K)

    [output]
    * dist: numpy.ndarray of shape (N)
    '''
    
    if USE_VGG16:
        K = feature.shape[0]
        sim = scipy.spatial.distance.cdist(train_features, feature.reshape(1, K))
    else:
        K = feature.shape[0]
        sim = scipy.spatial.distance.cdist(train_features, feature.reshape(1, K))
	
    return sim
