import cv2
import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt
import torch

from depthEstimation import DepthEstimator
from vehicleDetection import VehicleDetector

def drawDetections(img, detections):
    # detections is a pandas dataframe with columns: xmin, ymin, xmax, ymax, confidence, class, name
    detection_img = img.copy()
    # Draw detections and add depth values as a label to the rectangle
    for index, row in detections.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        # Round xmin, ymin, xmax, ymax to int
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cv2.rectangle(detection_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        depth_label = "Depth: {:.2f} m".format(round(row['depth'], 2))
        # Add black background panel to depth label
        label_size, _ = cv2.getTextSize(depth_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        panel_size = (label_size[0]+10, label_size[1]+10)
        panel_pos = (xmin, ymin - panel_size[1])
        cv2.rectangle(detection_img, panel_pos, (panel_pos[0]+panel_size[0], panel_pos[1]+panel_size[1]), (0,0,0), -1)
        cv2.putText(detection_img, depth_label, (panel_pos[0]+5, panel_pos[1]+label_size[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return detection_img

def getMeanDepth(depth, detections):
    # detections is a pandas dataframe with columns: xmin, ymin, xmax, ymax, confidence, class, name
    # Get depth of each detection
    # Add depth to detection data frame as a new column
    depth_per_detection = []
    for i, detection in detections.iterrows():
        xmin, ymin, xmax, ymax = detection["xmin"], detection["ymin"], detection["xmax"], detection["ymax"]
        # Round xmin, ymin, xmax, ymax to int
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        depth_values = depth[ymin:ymax, xmin:xmax].flatten()
        depth_values = depth_values[depth_values != 0] # remove zeros
        depth_per_detection.append(depth_values.mean())

    # Add depth to detection data frame as a new column
    detections["depth"] = depth_per_detection

    return detections



if __name__ == "__main__":
    frame_rate = 5 

    # Dummy camera intrinsic parameters
    image_width = 1280
    image_height = 720

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,20)
    fontScale              = 1
    fontColor              = (0,0,0)
    thickness              = 2 
    lineType               = 2

    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('test_images/vid1.mp4')
    cap.set(cv2.CAP_PROP_FPS, frame_rate)
    follower_depth_estimator = DepthEstimator(model_type='DPT_Large') # DPT_Large, MiDaS_small DPT_Large is more accurate but slower. MiDaS_small seems to be good enough though.
    vehicle_detector = VehicleDetector('./yolov5.pt', device='cpu')
    initial_frame = np.zeros((1280, 720, 3), dtype=np.uint8)

    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        ret, frame = cap.read()

        # Predict depth
        follower_depth_estimator.load_images_from_array([frame]) # Load first frame for testing
        depth_map = follower_depth_estimator.predict_depth()

        depth_map_norm = cv2.normalize(depth_map[0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colormap_depth_map = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)

        # Detect vehicles
        vehicle_boxes = vehicle_detector.detect(frame)
        vehicle_boxes = getMeanDepth(depth_map[0], vehicle_boxes)

        detection_img = drawDetections(frame, vehicle_boxes)
        #print(vehicle_boxes)
        cv2.imshow('detection_img', detection_img)

        #stack = np.hstack((frame, colormap_depth_map))
        #cv2.imshow('Frame and corresponding depth map', stack)

        baseline = 0

        cv2.putText(frame,f'Distance from initial frame: {baseline}',
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType)

    cap.release()
    cv2.destroyAllWindows()

