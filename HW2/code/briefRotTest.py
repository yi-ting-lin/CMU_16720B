import cv2
import BRIEF as BR
import numpy as np
import matplotlib.pyplot as plt

# test matches   
im1 = cv2.imread('../data/model_chickenbroth.jpg')
im2 = cv2.imread('../data/chickenbroth_01.jpg')


#perform rotation
h, w = im2.shape[:2]
center = (w//2, h//2)
angle_range = 30
scale = 1.0

angles = np.arange(0,angle_range,5)
matches_cnts = []
locs1, desc1 = BR.briefLite(im1)

for angle in angles:
    R = cv2.getRotationMatrix2D(center, angle, scale)
    im2 = cv2.warpAffine(im2, R, (w, h))
    locs2, desc2 = BR.briefLite(im2)
    matches = BR.briefMatch(desc1, desc2)
    matches_cnts.append(len(matches))

plt.plot(angles, matches_cnts)
plt.xlabel('rotation angles (degree)')
plt.ylabel('maches count')
plt.savefig('RotTest_plot.jpg')




#BR.plotMatches(im1,im2,matches,locs1,locs2)
