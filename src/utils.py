import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def video_to_frames(video_path, frames_path=None, frame_rate=1) -> list:
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

def load_frames(frames_path):
    frames = []
    # Load the frames in order
    for filename in os.listdir(frames_path):
        if filename.endswith(".jpg"):
            frame_path = os.path.join(frames_path, filename)
            frames.append(frame_path)
    # Sort the frames by their frame number
    frames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    loaded_frames = [cv2.imread(frame) for frame in frames]
    return loaded_frames



# Iterate through and show any frame array
def show_frames(frames: list):
    for i, frame in enumerate(frames):
        cv2.imshow(f"Frame {i}", frame)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def split_frames(frames: list, t: int):
    '''
    Takes an array of video frames and creates two separate arrays at time t away from each other.
    Input:
        frames: array of video frames
        t: decides how far apart the frames are in time(not necessarily in seconds, depending on the frame rate).
    Output:
        follower: array of frames that are t frames behind the lead camera
        lead: array of frames that are t frames ahead of the follower camera
    '''
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
def show_split_frames(follower: list, lead: list):
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

# Can extract features from a single frame or a batch of frames
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

# Draws the features onto the frames
def draw_keypoints(frames, keypoints):
    return [cv2.drawKeypoints(np.copy(frame), keypoints[i], np.copy(frame), color=(0, 255, 0), flags=0) for i, frame in enumerate(frames)]

# Match features between two frames or two sets of frames
def match_features(follower_kp, follower_desc, lead_kp, lead_desc, matcher_type='bf', ratio_thresh=0.75):
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
    
    # filter matches using Lowe's ratio test
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

    # Filter using RANSAC
    if len(good_matches) >= 4:
        homography, mask = cv2.findHomography(follower_pts, lead_pts, cv2.RANSAC, 7.0) 
        matches_mask = mask.ravel().tolist()
        good_matches = [m for i, m in enumerate(good_matches) if matches_mask[i]]

    return good_matches


# Does feature matching for a batch of frames
def match_feature_batch(follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors, matcher_type='bf', ratio_thresh=0.75):
    return [match_features(follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors, matcher_type=matcher_type, ratio_thresh=ratio_thresh) for follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors in zip(follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors)]


def visualize_matches(img1, keypoints1, img2, keypoints2, matches, show=False):
    vis_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=2)
    
    if show:
        cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Matches", (960, 480))
        cv2.imshow("Matches", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return vis_img

# Computes the fundamental matrix and uses cv.stereoRectifyUncalibrated to compute the rectification transforms
# https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/
def compute_rectification_transforms(img1, img2, kp1, kp2, matches):
    # Convert keypoint coordinates to numpy arrays
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.01, 0.9)

    # Only use inlier points, check that mask is not empty and we have more than 8 points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Compute the rectification transforms
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, imgSize=img1.shape[:2])

    return F, H1, H2, mask

def draw_epipolar_lines(img1, img2, kp1, kp2, matches):
    F, H1, H2, mask = compute_rectification_transforms(img1, img2, kp1, kp2, matches)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the epilines for the inlier points in img1 and img2
    lines1 = cv2.computeCorrespondEpilines(pts2, 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1, 1, F)
    lines2 = lines2.reshape(-1, 3)

    img5, img6 = drawlines(img1.copy(), img2.copy(), lines1, pts1, pts2)
    img3, img4 = drawlines(img2.copy(), img1.copy(), lines2, pts2, pts1)

    # Stack the images horizontally
    img_draw = np.hstack((img5, img3))
    img_draw = cv2.resize(img_draw, (img_draw.shape[1]//2, img_draw.shape[0]//2))
    cv2.namedWindow("Epipolar Lines", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Epipolar Lines", (img_draw.shape[1], img_draw.shape[0]))
    cv2.imshow("Epipolar Lines", img_draw)
    cv2.waitKey(0)

def rectify_images(img1, img2, kp1, kp2, matches):
    F, H1, H2, mask = compute_rectification_transforms(img1, img2, kp1, kp2, matches)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    img1_rectified = cv2.warpPerspective(img1, H1, (img1.shape[1], img1.shape[0]))
    img2_rectified = cv2.warpPerspective(img2, H2, (img2.shape[1], img2.shape[0]))

    # Compute the epilines for the inlier points in img1 and img2
    lines1 = cv2.computeCorrespondEpilines(pts2, 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1, 1, F)
    lines2 = lines2.reshape(-1, 3)

    img5, img6 = drawlines(img1.copy(), img2.copy(), lines1, pts1, pts2)
    img3, img4 = drawlines(img2.copy(), img1.copy(), lines2, pts2, pts1)

    # Apply the rectification transforms to the epilines
    img5_rect = cv2.warpPerspective(img5, H1, img5.shape[:2])
    img3_rect = cv2.warpPerspective(img3, H2, img3.shape[:2])

    # Stack the images horizontally
    img_draw = np.hstack((img5_rect, img3_rect))
    img_draw = cv2.resize(img_draw, (img_draw.shape[1]//2, img_draw.shape[0]//2))

    return img1_rectified, img2_rectified, img_draw

def rectify_images_batch(follower_imgs, lead_imgs, follower_kp, lead_kp, matches):
    follower_rectified, lead_rectified, epipolar_imgs = [], [], []
    for i in range(len(follower_imgs)):
        try:
            follower_rect, lead_rect, epipolar_img = rectify_images(follower_imgs[i], lead_imgs[i], follower_kp[i], lead_kp[i], matches[i])
            follower_rectified.append(follower_rect)
            lead_rectified.append(lead_rect)
            epipolar_imgs.append(epipolar_img)
        except:
            print("Error rectifying images")
            continue
    return follower_rectified, lead_rectified, epipolar_imgs

def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img1src.shape
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        rad1 = np.round(pt1[0]).astype("int")
        rad2 = np.round(pt2[0]).astype("int")
        img1src = cv2.line(img1src, (x0, y0), (x1, y1), color, 1)
        img1src = cv2.circle(img1src, tuple(rad1), 5, color, -1)
        img2src = cv2.circle(img2src, tuple(rad2), 5, color, -1)

    return img1src, img2src

# Estimate the pose from the essential matrix
def estimate_pose(follower_kp, lead_kp, matches, focal_length=1, principal_point=(0, 0)):
    '''
    follower_pts: points in the follower frame, we can get this from the matching function
    lead_pts: points in the lead frame, again we can get this from the matching functiona
    focal_length and principal_point: intrinsic parameters of the camera, might need to calibrate for this, unless we use videos from the lab cameras.
    '''
    # Convert keypoint coordinates to numpy arrays
    follower_pts = np.float32([follower_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    lead_pts = np.float32([lead_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    camera_matrix = np.array([[focal_length, 0, principal_point[0]], [0, focal_length, principal_point[1]], [0, 0, 1]], dtype=np.float32)

    E, mask = cv2.findEssentialMat(follower_pts, lead_pts, focal_length, principal_point, cv2.RANSAC, 0.999, 1.0)

    _, R, t, _ = cv2.recoverPose(E, follower_pts, lead_pts, camera_matrix, mask=mask)

    return R, t

#https://gist.github.com/davegreenwood/e1d2227d08e24cc4e353d95d0c18c914
def triangulate_nviews(P, ip):
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]

def triangulate_points(keypoints1, keypoints2, matches, cameraMatrix1=None, cameraMatrix2=None):
    # Extract the matched keypoints
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts1_u = cv2.convertPointsToHomogeneous(pts1)
    pts2_u = cv2.convertPointsToHomogeneous(pts1)

    if not len(pts1) == len(pts2):
        raise ValueError('Number of points and number of cameras not equal.')
    X = [triangulate_nviews([cameraMatrix1, cameraMatrix2], [pt[0], pt[1]]) for pt in zip(pts1_u, pts2_u)]
    return X

    return 

def triangulate_points_batch(follower_kp, lead_kp, matches, cameraMatrix1=None, cameraMatrix2=None):
    points3D = []
    for i in range(len(follower_kp)):
        points3D.append(triangulate_points(follower_kp[i], lead_kp[i], matches[i], cameraMatrix1, cameraMatrix2))
    return points3D

### EXAMPLE USAGE ###
if __name__ == "__main__":
    '''
    Observations:
        - ORB seems to give way fewer false matches than SIFT.
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
    frames = video_to_frames('./videos/vid1.mp4', './frames', frame_rate=3)
    # Or load frames from directory
    #frames = load_frames('./frames/horizontal_test_frames')

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
    #_, _, matches = match_features(follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors, matcher_type=MATCHER_TYPE, ratio_thresh=RATIOTHRESH)
    # Do matches for a batch of frames
    matches_per_frame = match_feature_batch(follower_keypoints, follower_descriptors, lead_keypoints, lead_descriptors, matcher_type=MATCHER_TYPE, ratio_thresh=RATIOTHRESH)

    # Visualize matches
    match_images = [visualize_matches(follower[i], follower_keypoints[i], lead[i], lead_keypoints[i], matches_per_frame[i]) for i in range(len(matches_per_frame))]
    show_frames(match_images)

    # Rectify images
    #_, _, rectified_images = rectify_images_batch(follower, lead, follower_keypoints, lead_keypoints, matches_per_frame)
    #show_frames(rectified_images)

    # Triangulate points
    points3D = triangulate_points_batch(follower_keypoints, lead_keypoints, matches_per_frame, cameraMatrix1=P1, cameraMatrix2=P2)
    # Flatten list of lists
    points3D = [item for sublist in points3D for item in sublist]

    idx = 5 # Testing for a single time step

    # Estimate the essential matrix between two frames
    matches = match_features(follower_keypoints[idx], follower_descriptors[idx], lead_keypoints[idx], lead_descriptors[idx], matcher_type=MATCHER_TYPE, ratio_thresh=RATIOTHRESH)
    R, t = estimate_pose(follower_keypoints[idx], lead_keypoints[idx], matches)
    #print(t)

