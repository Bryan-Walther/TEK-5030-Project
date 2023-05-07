So far I'm using yolov5 for detecting cars, and then estimating a depth map using MiDaS(needs to be swapped with ZoeDepth).
Based on the bounding boxes for each detection, we get the average depth from the depth map by averaging all the pixels in the depth map for each detection, this is already implemented.
A drawDetections function takes the detections and displays them using bounding boxes with their depth labels on top.

Next up:
    - Crop the vechicles from the bounding box
    - Do detection on the cropped images to look for license plate
    - Edge detection on the cropped license plate to get the dimensions.
    - Calculate the depth from the known dimensions and the image sizes, we might need to know the focal length of the camera for this.
    - Align the predicted depth map based on the known depths from all the license plates using least squares, this function is already implemented in DepthEstimation.py
