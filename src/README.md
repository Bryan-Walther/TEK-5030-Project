# Summary

This is a Python script that uses computer vision techniques to estimate the relative distance between two cameras.
Specifically, the script reads a video file, extracts frames, and identifies key features in each frame.
It then matches the features across frames, estimates the pose of the cameras, and finally calculates the distance between the cameras.
Currently this is done on frame pairs extracted from a single video, which can be extended to working directly from two separate camera feeds.
These frames are then offset by a constant "t" such that the follower camera and the lead camera are "t" time steps apart from each other.

The code consists of the following main functions:

- `estimate_baseline(frame_num, follower_keypoints, lead_keypoints, matches, K, focal_length, principal_point)`: This function is supposed to estimates the distance between two cameras for each frame in the input video. 
It first calculates the disparity between the matched features in each frame, and then uses the disparity and the estimated depth of the features to calculate the distance between them.
- `filter_estimates(baseline_estimates)`: This function applies a filter to the baseline estimates to remove outliers.


The script imports four submodules:

- `videoProcessor`: Extracts frames from a video file and processes them.
- `extractor`: Detects features in each frame.
- `matcher`: Matches the features across frames.
- `DepthEstimation`: Calculates the depth of each frame using monocular depth estimation networks in torch. (might be useful)

There is also the `utils.py` file that has various utility functions for visualisation, pose estimation, image rectification, etc.

# Current problems and what to do next
The main issue is that the `estimate_baseline()` is mostly just a place holder and needs to be correctly implemented.
This doesn't work for now because we don't have the depth of the features, which seems to be necessary to use the following equation: $${Z} = f\frac{b_x}{d} \rightarrow b_x = \frac{Zf}{d} $$
Where Z is our depth, d is the disparity, f is the focal length, and b is the baseline(distance between the cameras).
The problem is that for stereo imaging, it seems we need to know the baseline to estimate the depth, and vice versa.

So the next step would be to find a way to estimate the depth, so that we can use this formula to estimate the baseline.
One possible approach could be using monocular depth estimation, where we use a CNN to estimate the depth map per frame.
An implementation of this can be seen in `DepthEstimator`, which loads as pre-trained MiDaS model that outputs the estimated depth map.
One problem with this is that its slow if we don't use GPU, however with GPU it could be used real time.
Another issue is that it only estimates relative depth, so we won't get anything in meters.
This could still give us a usable relative distance between the cameras, but up to some unknown scale.

# Usage

To use the script, you need to provide the following inputs:

- `video_file_path`: Path to the video file.
- `K`: Camera calibration matrix
- `D`: Camera distortion coefficients
- `EXTRACTION_TYPE`: Which feature extractor to use, currently only "ORB" or "SIFT".
- `MATCHER_TYPE`: Matching algorithm to use, currently either "bf" or "flann".
- `t` : The offset for the lead camera vs the follower camera in the given video.
- `frame_rate` : How many frames per second do we want to include

Here is an example of how to use the script:

```python
from videoProcessor import VideoProcessor
from extractor import Extractor
from matcher import Matcher
from DepthEstimation import DepthEstimator

EXTRACTION_TYPE = 'SIFT'
MATCHER_TYPE = 'bf'
t = 1
frame_rate = 4 

# Camera intrinsic matrix
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
]) 
# Camera distortion coefficients
D = np.array(...)

# Convert video to frames
video_processor = VideoProcessor(video_path='./videos/dji_vid4.mp4', frames_path='./frames', frame_rate=frame_rate, t=t, movement_mode='parallel', K=K, D=D)
# Or load frames from directory
video_processor = VideoProcessor(video_path='./videos/vid1.mp4', frames_path='./frames', frame_rate=1, t=1, load_frames=True)

# Extract features from frame/s using specified descriptor type
follower_extractor = Extractor(video_processor.follower_frames)
follower_extractor.extract_features(descriptor_type=EXTRACTION_TYPE)

lead_extractor = Extractor(video_processor.lead_frames)
lead_extractor.extract_features(descriptor_type=EXTRACTION_TYPE)

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

baselines, _ = estimate_baseline(len(follower_frames), follower_keypoints, lead_keypoints, matcher.matches, K, focal_length, principal_point=(cx, cy))
```
