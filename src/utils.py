import cv2
import os
import numpy as np
import scipy as sp

def video_to_frames(video_path, frames_path=None, frame_rate=1):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = int(round(fps / frame_rate))

    frame_count = 0
    seq_num = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % frame_interval == 0:
            if frames_path:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                frames_folder = f"{frames_path}/{video_name}_frames"
                if not os.path.exists(frames_folder):
                    os.makedirs(frames_folder)
                frame_path = f"{frames_folder}/{video_name}_frame_{seq_num}.jpg"
                cv2.imwrite(frame_path, frame)
    
            frames.append(frame)
            seq_num += 1

    cap.release()
    cv2.destroyAllWindows()

    return frames

# Iterate through any frame array
def show_frames(frames):
    for i, frame in enumerate(frames):
        cv2.imshow(f"Frame {i}", frame)

        key = cv2.waitKey(0)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

# Takes an array of video frames and creates two separate arrays at time t away from each other.
# t decides how far apart the frames are in time(not necessarily in seconds, depending on the frame rate)
def split_frames(frames, t):
    num_frames = len(frames)

    follower = [] # This camera will be behind
    lead = [] # This camera will be ahead

    for i in range(num_frames - t):
        frame = frames[i]
        if i < t:
            follower.append(frame)
        else:
            lead.append(frame)
            follower.append(frame)

    lead.extend(frames[-t:])
    return follower, lead

# Iterate through the follower and lead frames side by side
def show_split_frames(follower, lead):
    num_follower = len(follower)
    num_lead = len(lead)

    cv2.namedWindow("Lead and Follower Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lead and Follower Camera", (960, 480))

    for i in range(min(num_follower, num_lead)):
        follower_frame = follower[i]
        lead_frame = lead[i]

        combined_frame = cv2.hconcat([follower_frame, lead_frame])
        cv2.imshow("Lead and Follower Camera", combined_frame)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def extract_features(frames, descriptor_type='ORB'):
    if descriptor_type == 'ORB':
        feature_detector = cv2.ORB_create()
    elif descriptor_type == 'SIFT':
        feature_detector = cv2.xfeatures2d.SIFT_create()
    elif descriptor_type == 'SURF':
        feature_detector = cv2.xfeatures2d.SURF_create()

    keypoints = []
    descriptors = []

    if isinstance(frames, list):
        for i in range(len(frames)):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            kp, desc = feature_detector.detectAndCompute(gray, None)
            keypoints.append(kp)
            descriptors.append(desc)
    else:
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        kp, desc = feature_detector.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(desc)

    return keypoints, descriptors
def draw_keypoints(frames, keypoints):
    output_frames = []

    for i in range(len(frames)):
        kp_frame = np.copy(frames[i])
        kp_frame = cv2.drawKeypoints(kp_frame, keypoints[i], kp_frame, color=(0, 255, 0), flags=0)

        output_frames.append(kp_frame)

    return output_frames

def match_features(follower_kp, follower_desc, lead_kp, lead_desc, matcher_type='bf', ransac_thresh=5.0):
    if matcher_type == 'bf':
        # initialize a Brute-Force Matcher
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif matcher_type == 'flann':
        # initialize a FLANN Matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError(f"Invalid matcher type: {matcher_type}")
    # TODO: Implement matching and RANSAC.

### EXAMPLE USAGE ###
if __name__ == "__main__":
    EXTRACTION_TYPE = 'ORB'
    MATCHER_TYPE = 'bf'
    
    # Convert video to frames
    frames = video_to_frames('./videos/vid1.mp4', './frames', frame_rate=1)

    # Split frames into follower and lead
    follower, lead = split_frames(frames, t=3)

    # Show follower and lead frames side by side
    #show_split_frames(follower, lead)

    # Extract features from frame/s using specified descriptor type
    follower_keypoints, follower_descriptors = extract_features(follower, descriptor_type=EXTRACTION_TYPE)
    lead_keypoints, lead_descriptors = extract_features(lead, descriptor_type=EXTRACTION_TYPE)

    # Use this to visualize the features
    show_split_frames(draw_keypoints(follower, follower_keypoints), draw_keypoints(lead, lead_keypoints))

    # Do matching between two frames
    #follower_kps, lead_kps, matches = match_features(follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors, matcher_type=MATCHER_TYPE)
