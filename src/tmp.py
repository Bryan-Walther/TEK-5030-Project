import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('frames/vid1_frames/vid1_frame_0.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('frames/vid1_frames/vid1_frame_2.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
print("dtype of des1: ", (des1.dtype))
print("dtype of kp2: ", type(kp2))
print("dtype of des2: ", type(des2))

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
