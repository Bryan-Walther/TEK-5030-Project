import cv2
import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt
import torch

from depthEstimation import DepthEstimator
from vehicleDetection import VehicleDetector
from plateDetection import PlateDetector

'''
These extra functions should be moved to a utils file later
'''
def drawVehicleDetections(img, detections):
    # detections is a pandas dataframe with columns: xmin, ymin, xmax, ymax, confidence, class, name
    # Draw detections and add depth values as a label to the rectangle
    detection_img = img.copy()
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

def cropDetection(src_img, detections, obj_type='vehicle'):
    # detections is a pandas dataframe with columns: xmin, ymin, xmax, ymax, confidence, class, name
    # Draw detections and add depth values as a label to the rectangle
    # Returns a list of tuples containing the cropped images and location of the cropped image in the original image
    if obj_type == 'vehicle':
        detection_img = src_img.copy()
        img = src_img
    else:
        detection_img = src_img[0].copy()
        img = src_img[0]
        

    cropped_images = []
    for i, detection in detections.iterrows():
        xmin, ymin, xmax, ymax = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
        # Round xmin, ymin, xmax, ymax to int
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        label = f"cropped_{i}"
        cv2.rectangle(detection_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(detection_img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cropped_img = img[ymin:ymax, xmin:xmax]
        location = (xmin, ymin, xmax, ymax)
        if obj_type == 'plate':
            org_xmin, org_ymin, org_xmax, org_ymax = src_img[1]
            location = (org_xmin + xmin, org_ymin + ymin, org_xmin + xmax, org_ymin + ymax)

        cropped_images.append((cropped_img, location))
    return cropped_images

def showCroppedDetection(cropped_images, label=''):
    # Uses cv2 to show all the detections in a grid on a single window
    num_cols = 4
    num_rows = -(-len(cropped_images) // num_cols)  # Ceiling division to calculate number of rows
    grid_size = 100
    padding = 10
    output_img = np.zeros((num_rows * (grid_size + padding) + padding, num_cols * (grid_size + padding) + padding, 3), dtype=np.uint8)
    output_img.fill(255)
    for i, cropped_img in enumerate(cropped_images):
        row = i // num_cols
        col = i % num_cols
        x = col * (grid_size + padding) + padding
        y = row * (grid_size + padding) + padding
        resized_img = cv2.resize(cropped_img[0], (grid_size, grid_size), interpolation=cv2.INTER_AREA)
        output_img[y:y+grid_size, x:x+grid_size] = resized_img
    cv2.imshow(f"Cropped Detections {label}", output_img)

def getEdges(img):
    # Convert image to grayscale
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the contour with the largest area
    max_contour = max(contours, key=cv2.contourArea)
    # Find the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(max_contour)
    # Draw the rectangle on the original image
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

    return img

# Draw an array of rectangles on an image (general purpose)
def draw(img, detections):
    if detections is None:
        return
    # Draw a rectangle for each detection on the copy image
    for detect in detections:
        # Extract the coordinates of the detection
        xmin, ymin, xmax, ymax = detect
        # Draw a rectangle around the detection on the copy image
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
    # Return the copy image with the rectangles drawn on it


if __name__ == "__main__":
    FRAME_RATE = 5 
    CONFIDENCE_THRESHOLD = 0.85 # Only show detections with confidence above this threshold

    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('test_images/vid1.mp4')
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
    follower_depth_estimator = DepthEstimator(model_type='MiDaS_small') # DPT_Large, MiDaS_small DPT_Large is more accurate but slower. MiDaS_small seems to be good enough though.
    vehicle_detector = VehicleDetector('./yolov5.pt', device='cpu', confidence_threshold=CONFIDENCE_THRESHOLD)
    plate_detector = PlateDetector(device='cpu')

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
        if vehicle_boxes is not None:
            vehicle_boxes = getMeanDepth(depth_map[0], vehicle_boxes) # Adds a depth column to the vehicle_boxes dataframe

            # Draw detections of vehicles
            vehicle_detection_img = drawVehicleDetections(frame, vehicle_boxes)

            # Show cropped detections
            vehicle_cropped_images = cropDetection(frame, vehicle_boxes, obj_type='vehicle')
            showCroppedDetection(vehicle_cropped_images, label='vehicles')
            for vehicle_cropped_img in vehicle_cropped_images:
                plate_boxes = plate_detector.detect(vehicle_cropped_img[0])
                if plate_boxes is not None:
                    plate_cropped_img = cropDetection(vehicle_cropped_img, plate_boxes, obj_type='plate')
                    plate_highlighted_img = [(getEdges(img), location) for img, location in plate_cropped_img]
                    showCroppedDetection(plate_highlighted_img, label='plates')
                    draw(vehicle_detection_img, [plate[1] for plate in plate_cropped_img]) # Overlay the plates onto the vehicle detection image
            
            
            cv2.imshow('detection_img', vehicle_detection_img)

        # Show frame and depth map side by side
        #stack = np.hstack((frame, colormap_depth_map))
        #cv2.imshow('Frame and corresponding depth map', stack)


    cap.release()
    cv2.destroyAllWindows()

