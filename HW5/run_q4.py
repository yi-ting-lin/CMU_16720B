import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap='gray')
    
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    
    
    # find the rows using..RANSAC, counting, clustering, etc.
    heights = [bbox[2] - bbox[0] for bbox in bboxes]
    min_height = min(heights)

    # (centerX, centerY, width, height)
    pos_list = [((bbox[3]+bbox[1])//2, (bbox[2]+bbox[0])//2, bbox[3]-bbox[1], bbox[2]-bbox[0]) for bbox in bboxes]

    # sort by centerY
    pos_list = sorted(pos_list, key=lambda x: x[1])
    
    rows = []
    row = []
    pre_y = None
    for pos in pos_list:
        if pre_y == None or pos[1] - pre_y < min_height: # still in the same row
            row.append(pos)
        else: #changing to next row
            #sorted by centerX and push into rows
            row = sorted(row,key=lambda x: x[0])
            rows.append(row)
            row = [pos]
        #update pre_y
        pre_y = pos[1]

    #sort andpush the last row
    row = sorted(row,key=lambda x: x[0])
    rows.append(row)


    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    raw_data = []
    for row in rows:
        row_raw_data = []
        for x, y, w, h in row:
            crop = bw[y-h//2:y+h//2, x-w//2:x+w//2]
            # pad it to square
            if h < w:
                w_pad = w//16
                h_pad = (w-h)//2+w_pad
            else:
                h_pad = h//16
                w_pad = (h-w)//2+h_pad
            crop = np.pad(crop, ((h_pad, h_pad), (w_pad, w_pad)), 'constant', constant_values=(1, 1))
            # resize to 32*32
            crop = skimage.transform.resize(crop, (32, 32))
            crop = skimage.morphology.erosion(crop, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
            crop = crop.T
            row_raw_data.append(crop.reshape(-1))
        raw_data.append(np.array(row_raw_data))
    
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    

    for row_data in raw_data:
        h1 = forward(row_data, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        output_str = ''
        for i in range(probs.shape[0]):
            predict_idx = np.argmax(probs[i])
            output_str += letters[predict_idx]

        print(output_str)

    plt.show()
    