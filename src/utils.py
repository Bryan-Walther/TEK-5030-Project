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

# Might be useful if we have several estimates with some outliers.
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
