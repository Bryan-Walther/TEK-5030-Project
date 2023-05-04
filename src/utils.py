import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Iterate through and show any frame array
def show_frames(frames: list):
    for i, frame in enumerate(frames):
        cv2.imshow(f"Frame {i}", frame)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def show_features_split(follower, lead):
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
    # Use cv2.stereoRectify instead after calibration when we have K and D matrix
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

    return F, img1_rectified, img2_rectified, img_draw

def rectify_images_batch(follower_imgs, lead_imgs, follower_kp, lead_kp, matches):
    f_mats, follower_rectified, lead_rectified, epipolar_imgs = [], [], [], []
    for i in range(len(follower_imgs)):
        try:
            F, follower_rect, lead_rect, epipolar_img = rectify_images(follower_imgs[i], lead_imgs[i], follower_kp[i], lead_kp[i], matches[i])
            follower_rectified.append(follower_rect)
            lead_rectified.append(lead_rect)
            epipolar_imgs.append(epipolar_img)
            f_mats.append(F)
        except:
            print("Error rectifying images")
            continue
    return f_mats, follower_rectified, lead_rectified, epipolar_imgs

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
def estimate_pose(follower_kp, lead_kp, matches, camera_matrix, focal_length, principal_point):
    '''
    follower_pts: points in the follower frame, we can get this from the matching function
    lead_pts: points in the lead frame, again we can get this from the matching functiona
    focal_length and principal_point: intrinsic parameters of the camera, might need to calibrate for this, unless we use videos from the lab cameras.
    '''
    # Convert keypoint coordinates to numpy arrays
    follower_pts = np.float32([follower_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    lead_pts = np.float32([lead_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

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

def estimate_baseline(frame_num, follower_keypoints, lead_keypoints, matches, K, focal_length, principal_point, depth_maps, scale_factor=1):
    baseline_per_frame = []
    normalized_pose_per_frame = []
    for idx in range(len(follower_keypoints)):
        baselines = []
        follower_depth_map = depth_maps[idx]

        # Calculate the disparity of the matched features between the two cameras
        for m in matches[idx]:
            x1, y1 = follower_keypoints[idx][m.queryIdx].pt
            x2, y2 = lead_keypoints[idx][m.trainIdx].pt

            disparity = abs(x2 - x1)
            if disparity == 0:
               continue
            depth = follower_depth_map[int(y1), int(x1)]
            baseline = (depth * disparity) / focal_length
            baselines.append(baseline)
        try:
            mean_baseline = np.mean(baselines)
            mean_baseline_filtered = filter_estimates(baselines)
        except:
            print('Could not calculate disparity for frame {}'.format(idx))
            continue

        # Get the translation vector from the relative pose between the two cameras
        # Pose of the lead camera relative to the follower camera (...I think, might be the other way around)
        R, T = estimate_pose(follower_keypoints[idx], lead_keypoints[idx], matches[idx], K, focal_length, principal_point=principal_point)

        baseline = mean_baseline

        baseline_filtered = mean_baseline_filtered
        baseline_per_frame.append(baseline_filtered)
        normalized_pose_per_frame.append((R, T))

        print("Baseline for time {} is {} ".format(idx, baseline * scale_factor))
        print("Baseline for time {} is {}  (FILTERED)".format(idx, baseline_filtered * scale_factor))
        print("Translation vector for time {} is {}".format(idx, T.T))
        print("\n")

    return baseline_per_frame, normalized_pose_per_frame
