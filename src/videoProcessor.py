import cv2
import matplotlib.pyplot as plt
import os
from depthEstimation import DepthEstimator

class VideoProcessor:
    def __init__(self, video_path=None, frames_path=None, frame_rate=1, load_frames=False):   
        if video_path is None and frames_path is None:
            raise Exception("Must provide either video_path or frames_path")
        self.video_path = video_path
        self.frames_path = frames_path
        self.frame_rate = frame_rate
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        if load_frames and frames_path:
            self.frames = self.load_frames()
        else:
            self.frames = self.video_to_frames()

        self.frame_width = self.frames[0].shape[1]
        self.frame_height = self.frames[0].shape[0]

        follower_depth_estimator = DepthEstimator(model_type='MiDaS_small') # DPT_Large, MiDaS_small DPT_Large is more accurate but slower. MiDaS_small seems to be good enough though.
        follower_depth_estimator.load_images_from_array(self.frames) # Load first frame for testing

        self.depth_maps = follower_depth_estimator.predict_depth()


    def video_to_frames(self) -> list:
        if not self.cap.isOpened():
            print("Error opening video file")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps / self.frame_rate))

        frame_count = 0
        seq_num = 0
        frames = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_interval == 0:
                if self.frames_path:
                    video_name = os.path.splitext(os.path.basename(self.video_path))[0]
                    frames_folder = f"{self.frames_path}/{video_name}_frames"
                    if not os.path.exists(frames_folder):
                        os.makedirs(frames_folder)
                    frame_path = f"{frames_folder}/{video_name}_frame_{seq_num}.jpg"
                    cv2.imwrite(frame_path, frame)
        
                frames.append(frame)
                seq_num += 1

        self.cap.release()
        cv2.destroyAllWindows()

        return frames

    def load_frames(self):
        frames = []
        # Load the frames in order
        for filename in os.listdir(self.frames_path):
            if filename.endswith(".jpg"):
                frame_path = os.path.join(self.frames_path, filename)
                frames.append(frame_path)
        # Sort the frames by their frame number
        frames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        loaded_frames = [cv2.imread(frame) for frame in frames]
        return loaded_frames

    def show_split_frames(self):
        num_frames = len(self.frames)
        num_depth_maps = len(self.depth_maps)

        # Show the frame and depth map side by side using matplotlib
        for i in range(num_frames):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Frame and Depth Map')
            ax1.imshow(self.frames[i])
            ax2.imshow(self.depth_maps[i])
            plt.show()

