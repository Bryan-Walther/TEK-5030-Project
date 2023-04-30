import cv2
import numpy as np
from utils import *

if __name__ == "__main__":
    '''
    Observations:
        - SIFT is a better than ORB, takes longer though.
        - Apparently, SURF is patented and no longer free to use.
        - Filtering with RANSAC improves the matching a lot.

    TODO:
        - Rectify function is broken after RANSAC filtering.
        - Write a camera calibration function to get the intrinsic parameters.
        - Extend estimate_pose function to calculate the scale factor which can give us the relative baseline/distance from the cameras
        - Try to write a function to help visualize the poses, maybe having a third window with overlaid poses?
        - Writing a bunch of visualization functions will probably help for the report.
        - Figure out what the commands at each time step t should look like. 
          Do we just take a step of magnitude k in the direction of the normalized pose, if the scale factor is greater than some threshold maybe?
        - Refactor: some of these functions should probably not be here.
        - (maybe, hopefully not) try to set up some simulation.
    '''
    EXTRACTION_TYPE = 'ORB'
    MATCHER_TYPE = 'bf'
    RATIOTHRESH = 0.99 #

    # Dummy camera matrices
    P1 = np.array([[5.010e+03, 0.000e+00, 3.600e+02, 0.000e+00],
               [0.000e+00, 5.010e+03, 6.400e+02, 0.000e+00],
               [0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00]])

    P2 = np.array([[5.037e+03, -9.611e+01, -1.756e+03, 4.284e+03],
                   [2.148e+02,  5.354e+03,  1.918e+02, 8.945e+02],
                   [3.925e-01,  7.092e-02,  9.169e-01, 4.930e-01]])
    
    # Convert video to frames
    frames = video_to_frames('./videos/vid2.mp4', './frames', frame_rate=4)
    # Or load frames from directory
    #frames = load_frames('./frames/horizontal_test_frames')

    # Split frames into follower and lead
    follower, lead = split_frames(frames, t=1)

    # Show follower and lead frames side by side
    #show_split_frames(follower, lead)

    # Extract features from frame/s using specified descriptor type
    follower_keypoints, follower_descriptors = extract_features(follower, descriptor_type=EXTRACTION_TYPE)
    lead_keypoints, lead_descriptors = extract_features(lead, descriptor_type=EXTRACTION_TYPE)

    # Use this to visualize the features
    show_split_frames(draw_keypoints(follower, follower_keypoints), draw_keypoints(lead, lead_keypoints))

    # Do matching between two frames
    #_, _, matches = match_features(follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors, matcher_type=MATCHER_TYPE, ratio_thresh=RATIOTHRESH)
    # Do matches for a batch of frames
    matches_per_frame = match_feature_batch(follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors, matcher_type=MATCHER_TYPE, ratio_thresh=RATIOTHRESH)

    # Visualize matches
    match_images = [visualize_matches(follower[i], follower_keypoints[i], lead[i], lead_keypoints[i], matches_per_frame[i]) for i in range(len(matches_per_frame))]
    show_frames(match_images)

    # Rectify images
    _, _, rectified_images = rectify_images_batch(follower, lead, follower_keypoints, lead_keypoints, matches_per_frame)
    show_frames(rectified_images)

    # Triangulate points
    #points3D = triangulate_points_batch(follower_keypoints, lead_keypoints, matches_per_frame, cameraMatrix1=P1, cameraMatrix2=P2)
    # Flatten list of lists
    #points3D = [item for sublist in points3D for item in sublist]

    # Estimate the essential matrix between two frames
    idx = 5 # Testing for a single time step
    matches = match_features(follower_keypoints[idx], follower_descriptors[idx], lead_keypoints[idx], lead_descriptors[idx], matcher_type=MATCHER_TYPE, ratio_thresh=RATIOTHRESH)
    R, t = estimate_pose(follower_keypoints[idx], lead_keypoints[idx], matches)
    print(t)

