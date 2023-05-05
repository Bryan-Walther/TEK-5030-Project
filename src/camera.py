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

    # Dummy camera intrinsic parameters
    image_width = 1280
    image_height = 720
    fov = 100.0  # field of view in degrees (114 for my camera) 94 degrees for dji phantom 4 footage
    sensor_width = 12.8333 # Sensor for DJI Phantom 4
    sensor_height = 7.2
    focal_length = (image_width / 2) / np.tan(np.deg2rad(fov / 2))

    fx = focal_length * sensor_width / image_width
    fy = focal_length * sensor_height / image_height

    cx = image_width / 2
    cy = image_height / 2

    # Camera intrinsic matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,20)
    fontScale              = 1
    fontColor              = (0,0,0)
    thickness              = 2 
    lineType               = 2

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, frame_rate)

    # Iterate over the frames
    follower_depth_estimator = DepthEstimator(model_type='MiDaS_small') # DPT_Large, MiDaS_small DPT_Large is more accurate but slower. MiDaS_small seems to be good enough though.


    initial_frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    initial = False
    while True:
        ret, frame = cap.read()

        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # Wait until space 
        if key == ord(' '):
            initial_frame = frame
            initial = True
            follower_extractor = Extractor([initial_frame])
            follower_extractor.extract_features(descriptor_type=EXTRACTION_TYPE)
            follower_keypoints, follower_descriptors, follower_frames = follower_extractor.get_params()
            follower_depth_estimator.load_images_from_array([initial_frame]) # Load first frame for testing
            follower_depths = follower_depth_estimator.predict_depth()




        if initial:
            leader_extractor = Extractor([frame])
            leader_extractor.extract_features(descriptor_type=EXTRACTION_TYPE)
            lead_keypoints, lead_descriptors, lead_frames = leader_extractor.get_params()

            matcher = Matcher(follower_extractor, leader_extractor, RANSAC=True)
            matcher.match_features(matcher_type=MATCHER_TYPE, ratio_thresh=RATIOTHRESH)

            match_images = matcher.visualize_matches()

            baselines, _ = estimate_baseline(1, follower_keypoints, lead_keypoints, matcher.matches, K, focal_length, principal_point=(cx, cy), depth_maps=follower_depths, scale_factor=SCALE_FACTOR)

            cv2.putText(match_images[0],f'Distance from initial frame: {baselines}',
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

            cv2.imshow('matches', match_images[0])






