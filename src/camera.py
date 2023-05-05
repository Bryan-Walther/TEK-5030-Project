import cv2
import numpy as np
from utils import *
from scipy.stats import iqr
import matplotlib.pyplot as plt

from videoProcessor import VideoProcessor
from extractor import Extractor
from matcher import Matcher
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
    EXTRACTION_TYPE = 'BRISK'
    SCALE_FACTOR = 10000 # Scale to multiply the predicted depth with. If we calibrate with some object of known size in the scene, we can extract the scale and get depth in meters.
    MATCHER_TYPE = 'bf'
    RATIOTHRESH = 0.59 # Not used anymore, but keeping it here just in case.
    frame_rate = 5 

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, frame_rate)

    # Iterate over the frames
    initial_frame = None
    while True:
        ret, frame = vid.read()
        key = cv2.waitKey(1)

        cv2.imshow('frame', frame)

        # Wait until space 
        if key == ord(' '):
            initial_frame = frame

        if initial_frame:
            # Initialize the feature extractor
            extractor = Extractor(EXTRACTION_TYPE)
            # Initialize the matcher
            matcher = Matcher(MATCHER_TYPE)
            # Initialize the depth estimator
            depthEstimator = DepthEstimator(videoProcessor, extractor, matcher, SCALE_FACTOR)

            # Get the depth map
            depth_map = depthEstimator.getDepthMap()

            # Show the depth map
            cv2.imshow('depth map', depth_map)

            # Wait until space 
            if key == ord(' '):
                initial_frame = frame
          






