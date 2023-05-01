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
        - The baseline calculation seems somewhat resonable if we at least know the fov of the camera and have good features.
            This seems to be the case with the dji drone video "dji_vid" using the correct fov of 94, as well as the image dimensions to set the principle point in the center.

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
    RATIOTHRESH = 0.79 #
    video_processor = VideoProcessor(video_path='./videos/dji_vid.mp4', frames_path='./frames', frame_rate=4, t=1)

    # Dummy camera intrinsic parameters
    image_width = video_processor.frame_width
    image_height = video_processor.frame_height
    fov = 94.0  # field of view in degrees (114 for my camera) 94 degrees for dji phantom 4 footage
    focal_length = (image_width / 2) / np.tan(np.deg2rad(fov / 2))
    cx = image_width / 2
    cy = image_height / 2

    # Camera intrinsic matrix
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ]) 
    
    # Convert video to frames
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
    # Calculate baseline (This seems to be somewhat reasonable for the dji footage with the dummy camera parameters)
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
            R, T = estimate_pose(follower_keypoints[idx], lead_keypoints[idx], matcher.matches[idx], focal_length=focal_length, principal_point=(cx, cy))
        except:
            continue
        T = T / T[2]  # Normalize translation vector
        T = T.reshape((3, 1))

        # Calculate baseline using the magnitude of the translation vector
        baseline = (focal_length * mean_disparity) / np.linalg.norm(T)
        baseline_per_frame.append(baseline)
        # Print baseline from mm to meters
        print("Baseline for time {} is {}".format(idx, baseline / 1000))
