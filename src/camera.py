import cv2
import numpy as np
from utils import *
from scipy.stats import iqr
import matplotlib.pyplot as plt

from videoProcessor import VideoProcessor
from depthEstimation import DepthEstimator

if __name__ == "__main__":
    '''
    Observations:
        - SIFT is a better than ORB, takes longer though.
        - ORB is the fastest, can result in few matches.
        - BRISK is somewhere in the middle in terms of performance and quality.
        - Apparently, SURF is patented and no longer free to use.
        - Filtering with RANSAC improves the matching a lot.

    TODO:
        - Write a camera calibration function to get the intrinsic parameters. (maybe not necessary)
        - Try to write a function to help visualize the poses, maybe having a third window with overlaid poses?
        - Writing a bunch of visualization functions will probably help for the report.
        - Rectify the images with respect to the calibrated camera parameters.
        - Do the baseline estimation with the stereo-rectified images and see if it improves the results.
        - (maybe, hopefully not) try to set up some simulation.
    '''
    frame_rate = 5 

    # Dummy camera intrinsic parameters
    image_width = 1280
    image_height = 720

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,20)
    fontScale              = 1
    fontColor              = (0,0,0)
    thickness              = 2 
    lineType               = 2

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, frame_rate)
    follower_depth_estimator = DepthEstimator(model_type='DPT_Large') # DPT_Large, MiDaS_small DPT_Large is more accurate but slower. MiDaS_small seems to be good enough though.

    initial_frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        ret, frame = cap.read()
        follower_depth_estimator.load_images_from_array([frame]) # Load first frame for testing

        depth_map = follower_depth_estimator.predict_depth()

        depth_map_norm = cv2.normalize(depth_map[0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colormap_depth_map = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)

        stack = np.hstack((frame, colormap_depth_map))
        cv2.imshow('Frame and corresponding depth map', stack)

        baseline = 0

        cv2.putText(frame,f'Distance from initial frame: {baseline}',
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType)

    cap.release()
    cv2.destroyAllWindows()







