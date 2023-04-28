import numpy as np
import cv2

# Define the camera matrices for each camera (intrinsic and extrinsic parameters)
# These matrices can be obtained by calibrating the cameras beforehand
camera_matrix_1 = np.array([...])
camera_matrix_2 = np.array([...])
dist_coeffs_1 = np.array([...])
dist_coeffs_2 = np.array([...])
R = np.array([...])
T = np.array([...])

# Capture the video inputs from the two cameras
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while True:
    # Read the frames from the two cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # Detect and match the features in the two frames
    feature_detector = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp1, des1 = feature_detector.detectAndCompute(frame1, None)
    kp2, des2 = feature_detector.detectAndCompute(frame2, None)
    matches = matcher.match(des1, des2)

    # Filter the matches based on their distance
    # This helps to remove outlier matches that may not be accurate
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = []
    for match in matches:
        if match.distance < 50:
            good_matches.append(match)

    # Compute essiential / fundemental matrix from correspondences

    # Show the frames with the matched points
    output_frame = cv2.drawMatches(frame1, kp1, frame2, kp2, good_matches, None)
    cv2.imshow("Output", output_frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video captures and close the windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
