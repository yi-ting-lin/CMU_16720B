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
    
    #load rects from section 3.1
    old_rects = np.load('carseqrects.npy')

    #initial ROI
    init_ROI = np.array([59, 116, 145, 151])
    ROI = init_ROI.copy()
    #p template and epsion to remove drift
    first_frame = car_seq[:, :, 0].copy()
    template_frame = car_seq[:, :, 0].copy()

    
    p_template = np.zeros(2)
    
    update_epsion = 0.4

    #stored rects
    rects = []
    rects.append(ROI)
    
    for i in range(1, total_frames):
        #First get nxt_frame
        cur_frame = car_seq[: , :, i]

        #get p frame LK algorithm and update ROI
        p = LucasKanade(template_frame, cur_frame, ROI)
        pn = p_template + p
        p_start = LucasKanade(first_frame, cur_frame, init_ROI, pn)
        new_ROI = [init_ROI[0]+p_start[0], init_ROI[1]+p_start[1], init_ROI[2]+p_start[0], init_ROI[3]+p_start[1]]

        #Create show images in color
        show_image = np.zeros((cur_frame.shape[0], cur_frame.shape[1], 3))
        show_image[:,:,0] = cur_frame.copy()
        show_image[:,:,1] = cur_frame.copy()
        show_image[:,:,2] = cur_frame.copy()
        

        #draw old (in yellow) and new rects (in gree)
        show_ROI_old    = [int(round(old_rects[i][0])), int(round(old_rects[i][1])), int(round(old_rects[i][2])), int(round(old_rects[i][3]))]
        show_ROI        = [int(round(new_ROI[0])), int(round(new_ROI[1])), int(round(new_ROI[2])), int(round(new_ROI[3]))]
        cv2.rectangle(show_image, (show_ROI_old[0], show_ROI_old[1]), (show_ROI_old[2], show_ROI_old[3]),color=(0,255,0))
        cv2.rectangle(show_image, (show_ROI[0], show_ROI[1]), (show_ROI[2], show_ROI[3]),color=(0,255,255))
        
        print('frame#' + str(i) +' finished processing.')
        #show image
        '''
        cv2.imshow('show image', show_image)
        cv2.waitKey(1)
        '''

        #Only update if norm(p_template - p) < update_epsion
        #if p_template is None or np.linalg.norm(p_template - p) < update_epsion:
        if np.linalg.norm(p_start - pn) < update_epsion:
            #print('hihi')
            template_frame = cur_frame
            p_template = p_start.copy()
            ROI = [init_ROI[0]+p_start[0], init_ROI[1]+p_start[1], init_ROI[2]+p_start[0], init_ROI[3]+p_start[1]]
        
        #store the images at i = 1, 100, 200,300 and 400
        if i == 1 or i == 100 or i == 200 or i == 300 or i == 400:
            cv2.imwrite(str(i) + '_wcrt.jpg', show_image*255)
        
        #store all the rects
        rects.append(new_ROI)


        #update p_pre
    #Store rects
    rects = np.array(rects)
    np.save('carseqrects-wcrt.npy', rects)


