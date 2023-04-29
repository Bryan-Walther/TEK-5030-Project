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
            gray = cv2.cvtColor(frames[i], cv2.IMREAD_GRAYSCALE)
            kp, desc = feature_detector.detectAndCompute(gray, None)
            keypoints.append(kp)
            descriptors.append(desc)
    else:
        gray = cv2.cvtColor(frames, cv2.IMREAD_GRAYSCALE)
        kp, desc = feature_detector.detectAndCompute(gray, None)
        return kp, desc

    return keypoints, descriptors

def draw_keypoints(frames, keypoints):
    output_frames = []

    for i in range(len(frames)):
        kp_frame = np.copy(frames[i])
        kp_frame = cv2.drawKeypoints(kp_frame, keypoints[i], kp_frame, color=(0, 255, 0), flags=0)

        output_frames.append(kp_frame)

    return output_frames

# Match features between two frames or two sets of frames
def match_features(follower_kp, follower_desc, lead_kp, lead_desc, matcher_type='bf'):
    if matcher_type == 'bf':
        # initialize a Brute-Force Matcher
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif matcher_type == 'flann':
        # initialize a FLANN Matcher
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError(f"Invalid matcher type: {matcher_type}")
    
    # match descriptors of the two images
    matches = matcher.match(follower_desc.astype(np.uint8), lead_desc.astype(np.uint8))
    
    # filter good matches using Lowe's ratio test
    ratio_thresh = 0.55
    good_matches = []
    for m in matches:
        if len(matches) > 1:
            if m.distance < ratio_thresh * matches[1].distance:
                good_matches.append(m)
        else:
            if m.distance < ratio_thresh * 500:
                good_matches.append(m)

    follower_pts = np.float32([follower_kp[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    lead_pts = np.float32([lead_kp[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    
    return follower_pts, lead_pts, good_matches

# Does feature matching for a batch of frames
def match_feature_batch(follower_kp, follower_desc, lead_kp, lead_desc, matcher_type='bf'):
    return [match_features(follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors, matcher_type=matcher_type)[2] for follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors in zip(follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors)]


def visualize_matches(img1, keypoints1, img2, keypoints2, matches, show=False):
    vis_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=2)
    
    if show:
        cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Matches", (960, 480))
        cv2.imshow("Matches", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return vis_img

### EXAMPLE USAGE ###
if __name__ == "__main__":
    '''
    Observations:

    - ORB seems to give way fewer false matches than SIFT.
    - Apparently, SURF is patented and no longer free to use.
    - Using flann matcher gives a larger number of matches than bf matcher,
      but does also increase the number of false matches.
    - I think flann could be better given a lower ratio threshold.
    '''
    EXTRACTION_TYPE = 'ORB' 
    MATCHER_TYPE = 'flann'
    
    # Convert video to frames
    frames = video_to_frames('./videos/vid1.mp4', './frames', frame_rate=1)

    # Split frames into follower and lead
    follower, lead = split_frames(frames, t=2)

    # Show follower and lead frames side by side
    #show_split_frames(follower, lead)

    # Extract features from frame/s using specified descriptor type
    follower_keypoints, follower_descriptors = extract_features(follower, descriptor_type=EXTRACTION_TYPE)
    lead_keypoints, lead_descriptors = extract_features(lead, descriptor_type=EXTRACTION_TYPE)

    # Use this to visualize the features
    #show_split_frames(draw_keypoints(follower, follower_keypoints), draw_keypoints(lead, lead_keypoints))

    # Do matching between two frames
    #_, _, matches = match_features(follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors, matcher_type=MATCHER_TYPE)
    # Do matches for a batch of frames
    matches_per_frame = match_feature_batch(follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors, matcher_type=MATCHER_TYPE)

    # Visualize matches
    match_images = [visualize_matches(follower[i], follower_keypoints[i], lead[i], lead_keypoints[i], matches_per_frame[i]) for i in range(len(matches_per_frame))]
    show_frames(match_images)
