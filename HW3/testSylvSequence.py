import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import cv2

from LucasKanade import LucasKanade
from LucasKanadeBasis import LucasKanadeBasis

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':

    #load basis
    bases = np.load('../data/sylvbases.npy')
    
    #Load frames
    car_seq = np.load('./../data/sylvseq.npy')
    total_frames = car_seq.shape[2]
    cur_frame = car_seq[:, :, 0]

    #initial ROI
    ROI         = np.array([101, 61, 155, 107])
    ROI_base    = np.array([101, 61, 155, 107])


    #p start and epsion to remove drift
    p_start = None

    #stored rects
    rects = []
    rects.append(ROI_base)
    
    for i in range(1, total_frames):
        pre_frame = car_seq[: , :, i-1]
        cur_frame = car_seq[: , :, i]

        #get p frame LK algorithm and update ROI
        p = LucasKanade(pre_frame, cur_frame, ROI)
        ROI = [ROI[0]+p[0], ROI[1]+p[1], ROI[2]+p[0], ROI[3]+p[1]]

        #get p base from LK base algorithm and cur_frame ROI
        p_base      = LucasKanadeBasis(pre_frame, cur_frame, ROI_base, bases)
        ROI_base    = [ROI_base[0]+p_base[0], ROI_base[1]+p_base[1], ROI_base[2]+p_base[0], ROI_base[3]+p_base[1]]

        #Create show images in color
        show_image = np.zeros((cur_frame.shape[0], cur_frame.shape[1], 3))
        show_image[:,:,0] = cur_frame.copy()
        show_image[:,:,1] = cur_frame.copy()
        show_image[:,:,2] = cur_frame.copy()
        show_ROI        = [int(round(ROI[0])), int(round(ROI[1])), int(round(ROI[2])), int(round(ROI[3]))]
        show_ROI_base   = [int(round(ROI_base[0])), int(round(ROI_base[1])), int(round(ROI_base[2])), int(round(ROI_base[3]))]
        cv2.rectangle(show_image, (show_ROI[0], show_ROI[1]), (show_ROI[2], show_ROI[3]),color=(0,255,0))
        cv2.rectangle(show_image, (show_ROI_base[0], show_ROI_base[1]), (show_ROI_base[2], show_ROI_base[3]),color=(0,255,255))
        
        print('frame#' + str(i) +' finished processing.')
        
        #show image
        '''
        cv2.imshow('show image', show_image)
        cv2.waitKey(1)
        '''

        #store the images at i = 1, 100, 200,350 and 400
        if i == 1 or i == 200 or i == 300 or i == 350 or i == 400:
            cv2.imwrite(str(i) + '_sylvseq.jpg', show_image*255)
        
        #store all the rects
        rects.append(ROI_base)
    
    #Store rects
    rects = np.array(rects)
    np.save('sylvseqrects.npy', rects)