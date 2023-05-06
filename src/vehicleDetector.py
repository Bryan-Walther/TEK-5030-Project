import torch
import matplotlib.pyplot as plt
import cv2

class VehicleDetector:
    def __init__(self, weights_path, device):
        self.weights_path = weights_path
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path).to(device)

    def detect(self, inp):
        pred = self.model(inp)
        # Return detections in pandas dataframe containing:
        # [xmin, ymin, xmax, ymax, confidence, class, name]
        return  pred.pandas().xyxy[0] 



# Loading in yolov5s - you can switch to larger models such as yolov5m or yolov5l, or smaller such as yolov5n
if __name__ == '__main__':
    detector = VehicleDetector('./yolov5.pt', 'cpu')

    # Run inference
    img_path = './test_images/img5.png'
    detections = detector.detect(img_path)

    print(detections)

    
    


