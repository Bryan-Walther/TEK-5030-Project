import cv2
import os

class VideoProcessor:
    def __init__(self, video_path, frames_path=None, frame_rate=1, t=1, load_frames=False):   
        self.video_path = video_path
        self.frames_path = frames_path
        self.frame_rate = frame_rate
        self.cap = cv2.VideoCapture(video_path)
        self.t = t
        if load_frames and frames_path:
            self.frames = self.load_frames()
        else:
            self.frames = self.video_to_frames()

        self.follower_frames, self.lead_frames = self.split_frames()

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

    def split_frames(self):
        '''
        Takes an array of video frames and creates two separate arrays at time t away from each other.
        Input:
            frames: array of video frames
            t: decides how far apart the frames are in time(not necessarily in seconds, depending on the frame rate).
        Output:
            follower: array of frames that are t frames behind the lead camera
            lead: array of frames that are t frames ahead of the follower camera
        '''
        num_frames = len(self.frames)

        follower = [] # This camera will be behind
        lead = [] # This camera will be ahead

        for i in range(num_frames - self.t):
            frame = self.frames[i]
            if i < self.t:
                follower.append(frame)
            else:
                lead.append(frame)
                follower.append(frame)

        lead.extend(self.frames[-self.t:])
        return follower, lead

    # Iterate through the follower and lead frames side by side
    def show_split_frames(self):
        num_follower = len(self.follower_frames)
        num_lead = len(self.lead_frames)

        cv2.namedWindow("Lead and Follower Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Lead and Follower Camera", (960, 480))

        for i in range(min(num_follower, num_lead)):
            follower_frame = self.follower_frames[i]
            lead_frame = self.lead_frames[i]

            combined_frame = cv2.hconcat([follower_frame, lead_frame])
            cv2.imshow("Lead and Follower Camera", combined_frame)
            key = cv2.waitKey(0)

            if key == ord('q'):
                break

        cv2.destroyAllWindows()
