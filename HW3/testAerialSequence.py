import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import cv2
from SubtractDominantMotion import SubtractDominantMotion
from InverseCompositionAffine import InverseCompositionAffine

# write your script here, we recommend the above libraries for making your animation
 #Create result directory
if __name__ == '__main__':
    
    #Load frames
    aerialseq = np.load('./../data/aerialseq.npy')
    total_frames = aerialseq.shape[2]
    
    #store mask
    masks = []

    for i in range(1, total_frames):
        pre_frame = aerialseq[:, :, i-1]
        cur_frame = aerialseq[:, :, i]
        
        #Make show_image as a color image
        show_image = np.zeros((cur_frame.shape[0], cur_frame.shape[1], 3))
        show_image[:,:,0] = cur_frame.copy()
        show_image[:,:,1] = cur_frame.copy()
        show_image[:,:,2] = cur_frame.copy()
        
        #Get the mask
        mask = SubtractDominantMotion(pre_frame, cur_frame)
        show_image[:,:,0][mask] = 255

        print('frame#' + str(i) +' finished processing.')
        #show the image
        '''
        cv2.imshow('show_image', show_image)
        cv2.waitKey(10)
        '''
        
        #store the images at i = 1, 100, 200,300 and 400
        if i == 30 or i == 60 or i == 90 or i == 120:
            cv2.imwrite(str(i) + '_aerialseq.jpg', show_image*255)
            masks.append(mask)


    masks = np.array(masks)
    np.save('aerialseqmasks.npy', masks)