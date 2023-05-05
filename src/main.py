import cv2
import numpy as np
from utils import *
from scipy.stats import iqr
import matplotlib.pyplot as plt

from videoProcessor import VideoProcessor
from depthEstimation import DepthEstimator

if __name__ == "__main__":
    frame_rate = 1 

    # Convert video to frames, and estimate depth map per frame
    video_processor = VideoProcessor(video_path='./videos/dji_vid4.mp4', frames_path='./frames', frame_rate=frame_rate)
    # Shows the frame and depth map side by side
    video_processor.show_split_frames() 


