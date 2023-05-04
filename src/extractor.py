import cv2
import numpy as np

class Extractor:
    def __init__(self, frames):
        self.frames = frames
        self.keypoints = None 
        self.descriptors = None 

    # Can extract features from a single frame or a batch of frames
    def extract_features(self, descriptor_type='ORB'):
        if descriptor_type == 'ORB':
            feature_detector = cv2.ORB_create()
        elif descriptor_type == 'SIFT':
            feature_detector = cv2.xfeatures2d.SIFT_create()
        elif descriptor_type == 'SURF':
            feature_detector = cv2.xfeatures2d.SURF_create()
        elif descriptor_type == 'BRISK':
            feature_detector = cv2.BRISK_create()

        if isinstance(self.frames, list):
            self.keypoints, self.descriptors = [], []
            for i in range(len(self.frames)):
                gray = cv2.cvtColor(self.frames[i], cv2.IMREAD_GRAYSCALE)
                kp, desc = feature_detector.detectAndCompute(gray, None)
                self.keypoints.append(kp)
                self.descriptors.append(desc)
        else:
            gray = cv2.cvtColor(self.frames, cv2.IMREAD_GRAYSCALE)
            self.keypoints, self.descriptors = feature_detector.detectAndCompute(gray, None)

    # Draws the features onto the frames
    def draw_keypoints(self):
        if self.keypoints is None:
            print('No keypoints to draw')
            return
        else:
            return [cv2.drawKeypoints(np.copy(frame), self.keypoints[i], np.copy(frame), color=(0, 255, 0), flags=0) for i, frame in enumerate(self.frames)]

    def get_params(self):
        return self.keypoints, self.descriptors, self.frames

