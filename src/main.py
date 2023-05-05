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
    t = 1 # Time step offset between the two cameras
    frame_rate = 5 

    # Convert video to frames
    video_processor = VideoProcessor(video_path='./videos/dji_vid4.mp4', frames_path='./frames', frame_rate=frame_rate, t=t, movement_mode='parallel')

    video_processor.follower_frames = [video_processor.follower_frames[0]] * len(video_processor.lead_frames)*2
    # Add lead frames reversed at the end of lead frames
    video_processor.lead_frames = video_processor.lead_frames + video_processor.lead_frames[::-1]

    # Or load frames from directory
    #video_processor = VideoProcessor(video_path='./videos/vid1.mp4', frames_path='./frames', frame_rate=1, t=1, load_frames=True)

    # Load depth estimator model and the frames to estimate depth from
    follower_depth_estimator = DepthEstimator(model_type='MiDaS_small') # DPT_Large, MiDaS_small DPT_Large is more accurate but slower. MiDaS_small seems to be good enough though.
    follower_depth_estimator.load_images_from_array(video_processor.follower_frames) # Load first frame for testing
    
    # Show follower and lead frames side by side
    video_processor.show_split_frames()

    # Dummy camera intrinsic parameters
    image_width = video_processor.frame_width
    image_height = video_processor.frame_height
    fov = 94.0  # field of view in degrees (114 for my camera) 94 degrees for dji phantom 4 footage
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

    # Extract features from frame/s using specified descriptor type
    follower_extractor = Extractor(video_processor.follower_frames)
    follower_extractor.extract_features(descriptor_type=EXTRACTION_TYPE)

    lead_extractor = Extractor(video_processor.lead_frames)
    lead_extractor.extract_features(descriptor_type=EXTRACTION_TYPE)

    # Use this to visualize the features
    show_features_split(follower_extractor.draw_keypoints(), lead_extractor.draw_keypoints())

    matcher = Matcher(follower_extractor, lead_extractor, RANSAC=True)
    matcher.match_features(matcher_type=MATCHER_TYPE, ratio_thresh=RATIOTHRESH)

    # Visualize matches
    match_images = matcher.visualize_matches()
    show_frames(match_images)

    # Rectify images
    follower_keypoints, follower_descriptors, follower_frames = follower_extractor.get_params()
    lead_keypoints, lead_descriptors, lead_frames = lead_extractor.get_params()
    f_mats, rectified_followers, rectified_lead, rectified_images = rectify_images_batch(follower_frames, lead_frames, follower_keypoints, lead_keypoints, matcher.matches)
    show_frames(rectified_images)
    
    # Predict depth for each follower frame
    follower_depths = follower_depth_estimator.predict_depth()

    # Show the depth maps
    for depth_map in follower_depths:
        plt.figure()
        plt.imshow(depth_map)
        plt.show()

    print("Calculating baseline for all frames without rectification\n")   
    baselines, _ = estimate_baseline(len(follower_frames), follower_keypoints, lead_keypoints, matcher.matches, K, focal_length, principal_point=(cx, cy), depth_maps=follower_depths, scale_factor=SCALE_FACTOR)
    print(f'Mean baseline for t={t} is {np.mean(baselines)*SCALE_FACTOR}')

    # Doing stereo rectification, re-matching and detecting features on rectified images, and calculating "baseline".
    '''
    # Extract feats and match again on rectified images
    follower_extractor_rect = Extractor(rectified_followers)
    follower_extractor_rect.extract_features(descriptor_type=EXTRACTION_TYPE)

    lead_extractor_rect = Extractor(rectified_lead)
    lead_extractor_rect.extract_features(descriptor_type=EXTRACTION_TYPE)

    matcher_rect = Matcher(follower_extractor_rect, lead_extractor_rect, RANSAC=True)
    matcher_rect.match_features(matcher_type=MATCHER_TYPE, ratio_thresh=RATIOTHRESH)

    # Visualize matches rectified
    match_images_rect = matcher_rect.visualize_matches()
    show_frames(match_images_rect)

    # Get rectified keypoints
    follower_keypoints_rect, _, _ = follower_extractor_rect.get_params()
    lead_keypoints_rect, _, _ = lead_extractor_rect.get_params()

    print("Calculating baseline for all frames after rectification\n")   
    baselines_rect = estimate_baseline(len(rectified_followers), follower_keypoints_rect, lead_keypoints_rect, matcher_rect.matches, K, focal_length, principal_point=(cx, cy))
    '''
