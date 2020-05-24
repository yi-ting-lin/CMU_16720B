import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image

#reference: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    #denoise
    image = skimage.restoration.denoise_bilateral(image, multichannel=True)

    #grayscale
    gray = skimage.color.rgb2gray(image)

    #threshold & morphology
    thresh = skimage.filters.threshold_otsu(gray)
    bw = skimage.morphology.closing(gray < thresh, skimage.morphology.square(7))

    #label
    label_image = skimage.morphology.label(bw, connectivity=2)
    properties = skimage.measure.regionprops(label_image)

    #remove invalid bbox
    mean_area = 0
    for p in properties:
        mean_area += p.area
    mean_area /= len(properties)

    for p in properties:
        if p.area < 2*mean_area and p.area > 0.5*mean_area:
            bboxes.append(p.bbox)

    bw = (~bw)
    bw = bw.astype(np.float)
   
    return bboxes, bw