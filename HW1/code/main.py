import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage

if __name__ == '__main__':
    num_cores = util.get_num_CPU()

    #path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    path_img = "../data/aquarium/sun_awravelrqnwyxbzm.jpg"
    
    image = skimage.io.imread(path_img)
    image[:,:,:] = image[:,:,:3]
    image = image.astype('float')/255
    labImage = skimage.color.rgb2lab(image)

    

    visual_words.compute_dictionary(num_workers=num_cores)

    #dictionary = np.load('dictionary.npy')
    #dictionary = np.load('dictionary_full.npy') 

    #util.display_filter_responses(filter_responses)
    #util.save_wordmap(word_map, 'wordmap.jpg')
    #visual_recog.get_feature_from_wordmap(word_map,len(dictionary))
    visual_recog.build_recognition_system(num_workers=num_cores)
    visual_recog.evaluate_recognition_system(num_workers=num_cores)

    #conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    #print(conf)
    #print(np.diag(conf).sum()/conf.sum())

    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    vgg16.eval()
    #deep_recog.test(vgg16)
    deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)
    #print(vgg16)
    conf = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
    #print(conf)
    #print(np.diag(conf).sum()/conf.sum())

