import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import cv2


# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    
    #Load frames
    car_seq = np.load('./../data/carseq.npy')
    total_frames = car_seq.shape[2]
    
    #initial ROI
    ROI = np.array([59, 116, 145, 151])

    #stored rects
    rects = []
    rects.append(ROI)
    
    for i in range(1, total_frames):
        #First get nxt_frame
        pre_frame = car_seq[:, :, i-1]
        cur_frame = car_seq[: , :, i]

        #get p frame LK algorithm and update ROI
        p = LucasKanade(pre_frame, cur_frame, ROI)
        ROI = [ROI[0]+p[0], ROI[1]+p[1], ROI[2]+p[0], ROI[3]+p[1]]
        
        #Create show images in color
        show_image = np.zeros((cur_frame.shape[0], cur_frame.shape[1], 3))
        show_image[:,:,0] = cur_frame.copy()
        show_image[:,:,1] = cur_frame.copy()
        show_image[:,:,2] = cur_frame.copy()
        show_ROI = [int(round(ROI[0])), int(round(ROI[1])), int(round(ROI[2])), int(round(ROI[3]))]
        cv2.rectangle(show_image, (show_ROI[0], show_ROI[1]), (show_ROI[2], show_ROI[3]),color=(0,255,255))
        
        print('frame#' + str(i) +' finished processing.')
        #show images
        '''
        cv2.imshow('show image', show_image)
        cv2.waitKey(1)
        '''

        #store the images at i = 1, 100, 200,300 and 400
        if i == 1 or i == 100 or i == 200 or i == 300 or i == 400:
            cv2.imwrite(str(i) + '.jpg', show_image*255)
        
        #store all the rects
        rects.append(ROI)
    
    #Store rects
    rects = np.array(rects)
    np.save('carseqrects.npy', rects)


