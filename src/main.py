import cv2
import numpy as np
from utils import *
from videoProcessor import VideoProcessor
from extractor import Extractor
from matcher import Matcher

if __name__ == "__main__":
    '''
    Observations:
        - SIFT is a better than ORB, takes longer though.
        - Apparently, SURF is patented and no longer free to use.
        - Filtering with RANSAC improves the matching a lot.

    TODO:
        - Write a camera calibration function to get the intrinsic parameters. (maybe not necessary)
        - Try to write a function to help visualize the poses, maybe having a third window with overlaid poses?
        - Writing a bunch of visualization functions will probably help for the report.
        - Given the intrinsic parameters and the relative pose between the cameras, calculate the baseline/distance.
        - Depending on the results, we might want to try bundle adjustment to improve things further.
        - (maybe, hopefully not) try to set up some simulation.
    '''
    EXTRACTION_TYPE = 'SIFT'
    MATCHER_TYPE = 'bf'
    RATIOTHRESH = 0.9 #

    # Dummy camera matrices
    P1 = np.array([[5.010e+03, 0.000e+00, 3.600e+02, 0.000e+00],
               [0.000e+00, 5.010e+03, 6.400e+02, 0.000e+00],
               [0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00]])

    P2 = np.array([[5.037e+03, -9.611e+01, -1.756e+03, 4.284e+03],
                   [2.148e+02,  5.354e+03,  1.918e+02, 8.945e+02],
                   [3.925e-01,  7.092e-02,  9.169e-01, 4.930e-01]])
    
    # Convert video to frames
    video_processor = VideoProcessor(video_path='./videos/horizontal_test.mp4', frames_path='./frames', frame_rate=3, t=1)
    # Or load frames from directory
    #video_processor = VideoProcessor(video_path='./videos/vid1.mp4', frames_path='./frames', frame_rate=1, t=1, load_frames=True)

    # Show follower and lead frames side by side
    #video_processor.show_split_frames()

    # Extract features from frame/s using specified descriptor type
    follower_extractor = Extractor(video_processor.follower_frames)
    follower_extractor.extract_features(descriptor_type=EXTRACTION_TYPE)

    lead_extractor = Extractor(video_processor.lead_frames)
    lead_extractor.extract_features(descriptor_type=EXTRACTION_TYPE)

    # Use this to visualize the features
    show_features_split(follower_extractor.draw_keypoints(), lead_extractor.draw_keypoints())

    matcher = Matcher(follower_extractor, lead_extractor)
    matcher.match_features(matcher_type=MATCHER_TYPE, ratio_thresh=RATIOTHRESH)

    # Visualize matches
    match_images = matcher.visualize_matches()
    show_frames(match_images)

    # Rectify images
    follower_keypoints, follower_descriptors, follower_frames = follower_extractor.get_params()
    lead_keypoints, lead_descriptors, lead_frames = lead_extractor.get_params()
    _, _, rectified_images = rectify_images_batch(follower_frames, lead_frames, follower_keypoints, lead_keypoints, matcher.matches)
    show_frames(rectified_images)

    # Triangulate points
    #points3D = triangulate_points_batch(follower_keypoints, lead_keypoints, matcher.matches, cameraMatrix1=P1, cameraMatrix2=P2)
    # Flatten list of lists
    #points3D = [item for sublist in points3D for item in sublist]

    # Move this to a util function
    # Calculate baseline (This is maybe wrong or at least not very accurate, especially without calibration)
    focal_length = P1[0, 0]  # Assuming both cameras have the same focal length
    # get baseline for all frames
    baseline_per_frame = []
    for idx in range(len(follower_frames)):
        displacements = []
        
        # Calculate the disparity of the matched features between the two cameras
        for m in matcher.matches[idx]:
            x1, y1 = follower_keypoints[idx][m.queryIdx].pt
            x2, y2 = lead_keypoints[idx][m.trainIdx].pt
            disparity = abs(x2 - x1)
            displacements.append(disparity)

        # Use the median or mean of the displacements as the baseline?
        #mean_disparity = np.median(displacements)
        mean_disparity = np.mean(displacements)

        # Get the translation vector from the relative pose between the two cameras
        try:
            R, T = estimate_pose(follower_keypoints[idx], lead_keypoints[idx], matcher.matches[idx], focal_length=focal_length, principal_point=(P1[0, 2], P1[1, 2]))
        except:
            continue
        T = T / T[2]  # Normalize translation vector
        T = T.reshape((3, 1))

        # Calculate baseline using the magnitude of the translation vector
        baseline = (focal_length * mean_disparity) / np.linalg.norm(T)
        baseline_per_frame.append(baseline)
        # Print baseline from mm to meters
        print("Baseline for time {} is {}".format(idx, baseline / 1000))
