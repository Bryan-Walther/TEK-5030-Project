import cv2
import numpy as np
from utils import *
from scipy.stats import iqr
import matplotlib.pyplot as plt

from videoProcessor import VideoProcessor
from extractor import Extractor
from matcher import Matcher
from DepthEstimation import DepthEstimator

def filter_estimates(baseline_estimates):
    baseline_estimates = np.array(baseline_estimates)

    median = np.median(baseline_estimates)
    q1, q3 = np.percentile(baseline_estimates, [25, 75])
    iqr = q3 - q1

    threshold = 1.5 * iqr
    inliers = baseline_estimates[np.abs(baseline_estimates - median) < threshold]
    print("Inliers: ", len(inliers))
    baseline_mean = np.mean(inliers)

    return baseline_mean
'''
About estimating the scale of the baseline(distance between cameras):
    This function doesn't actually calculate the baseline, the formula is wrong.
    This just calculates the disparity between the two cameras for all the points.
    It then takes the average of the disparities and returns that.
    Calculating the baseline using the correct formula requires us to know the depth of the points, and to estimate the depth using stereo, we need to know the baseline.
    Therefore, estimating the baseline this way does not seem possible, not sure.
    We could maybe use RGBD cameras? That way we would have the depth for each pixel, and we could directly calculate the baseline with the correct formula.
'''
def estimate_baseline(frame_num, follower_keypoints, lead_keypoints, matches, K, focal_length, principal_point):
    baseline_per_frame = []
    normalized_pose_per_frame = []
    for idx in range(len(rectified_followers)):
        displacements = []
        
        # Calculate the disparity of the matched features between the two cameras
        for m in matches[idx]:
            x1, y1 = follower_keypoints[idx][m.queryIdx].pt
            x2, y2 = lead_keypoints[idx][m.trainIdx].pt
            disparity = abs(x2 - x1)
            displacements.append(disparity)

        # Use the median or mean of the displacements as the baseline?
        # median is probably more robust to outliers
        #mean_disparity = np.median(displacements)
        try:
            mean_disparity = np.mean(displacements)
            mean_disparity_filtered = filter_estimates(displacements)
        except:
            print('Could not calculate disparity for frame {}'.format(idx))
            continue

        # Get the translation vector from the relative pose between the two cameras
        try:
            # Pose of the lead camera relative to the follower camera (...I think, might be the other way around)
            R, T = estimate_pose(follower_keypoints[idx], lead_keypoints[idx], matches[idx], K, focal_length, principal_point=(cx, cy))
        except:
            print('Could not estimate pose for frame {}'.format(idx))
            continue

        # Calculate baseline using the magnitude of the translation vector
        baseline = (focal_length * mean_disparity)
        # Actual baseline is (depth * disparity) / focal_length but we don't have depth
        baseline_filtered = (focal_length * mean_disparity_filtered) 
        baseline_per_frame.append(baseline_filtered)
        normalized_pose_per_frame.append((R, T))
        # round baseline to 3 decimal places
        print("Baseline for time {} is {} ".format(idx, round(baseline / 1000, 3)))
        print("Baseline for time {} is {}  (FILTERED)".format(idx, round(baseline_filtered / 1000, 3)))
        print("Translation vector for time {} is {}".format(idx, T.T))
        print("\n")

    return baseline_per_frame, normalized_pose_per_frame

if __name__ == "__main__":
    '''
    Observations:
        - SIFT is a better than ORB, takes longer though.
        - Apparently, SURF is patented and no longer free to use.
        - Filtering with RANSAC improves the matching a lot.
        - Don't think the baseline calculation is correct, its not calculating that, but still seems to be a good measure of relative distance.
        - The "baseline" calculation seems somewhat resonable if we at least know the fov of the camera and have good features.
            This seems to be the case with the dji drone video "dji_vid" using the correct fov of 94, as well as the image dimensions to set the principle point in the center.
        - The "baseline" estimations seem more consistent when skipping ransac filtering on the matches, but filtering the baseline estimates instead.
             Might be because RANSAC also removes a lot of false negatives, and without it, we get 10 times more estimates to work with which we can then filter.
             Hard to say if these values are more accurate though without ground truth.
        - RANSAC wasn't the problem, it was Lowe's ratio test. Removing it and keeping RANSAC gives way better results.
            The baseline estimates are now very consistent, even without filtering.

    TODO:
        - Write a camera calibration function to get the intrinsic parameters. (maybe not necessary)
        - Try to write a function to help visualize the poses, maybe having a third window with overlaid poses?
        - Writing a bunch of visualization functions will probably help for the report.
        - Rectify the images with respect to the calibrated camera parameters.
        - Do the baseline estimation with the rectified points.
        - Depending on the results, we might want to try bundle adjustment to improve things further.
        - (maybe, hopefully not) try to set up some simulation.
    '''
    EXTRACTION_TYPE = 'SIFT'
    MATCHER_TYPE = 'bf'
    RATIOTHRESH = 0.59 # Not used anymore, but keeping it here just in case.
    t = 1 # Time step offset between the two cameras
    frame_rate = 4 

    # Convert video to frames
    video_processor = VideoProcessor(video_path='./videos/dji_vid4.mp4', frames_path='./frames', frame_rate=frame_rate, t=t, movement_mode='parallel')
    # Or load frames from directory
    #video_processor = VideoProcessor(video_path='./videos/vid1.mp4', frames_path='./frames', frame_rate=1, t=1, load_frames=True)

    # Show follower and lead frames side by side
    #video_processor.show_split_frames()

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

    print("Calculating baseline for all frames without rectification\n")   
    baselines, _ = estimate_baseline(len(follower_frames), follower_keypoints, lead_keypoints, matcher.matches, K, focal_length, principal_point=(cx, cy))

    print(f'Mean baseline for t={t} is {round(np.mean(baselines) / 1000, 3)}')

    # Estimate Depth using a CNN from DepthEstimator.py, may or may not be useful.
    '''
    follower_depth_estimator = DepthEstimator()
    lead_depth_estimator = DepthEstimator()
    # Load images
    follower_depth_estimator.load_images_from_array(follower_frames)
    lead_depth_estimator.load_images_from_array(lead_frames)
    # Predict batch of images
    follower_depths = follower_depth_estimator.predict_depth()
    lead_depths = lead_depth_estimator.predict_depth()
    # Show depth maps
    plt.imshow(follower_depths[0])
    plt.show()
    '''

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

    # Triangulate points
    #points3D = triangulate_points_batch(follower_keypoints, lead_keypoints, matcher.matches, cameraMatrix1=P1, cameraMatrix2=P2)
    # Flatten list of lists
    #points3D = [item for sublist in points3D for item in sublist]
    '''
