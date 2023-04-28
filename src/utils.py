import cv2
import os

def video_to_frames(video_path, frames_path=None, frame_rate=1):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = int(round(fps / frame_rate))

    frame_count = 0
    seq_num = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % frame_interval == 0:
            if frames_path:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                frames_folder = f"{frames_path}/{video_name}_frames"
                if not os.path.exists(frames_folder):
                    os.makedirs(frames_folder)
                frame_path = f"{frames_folder}/{video_name}_frame_{seq_num}.jpg"
                cv2.imwrite(frame_path, frame)
    
            frames.append(frame)
            seq_num += 1

    cap.release()
    cv2.destroyAllWindows()

    return frames

# Iterate through any frame array
def show_frames(frames):
    for i, frame in enumerate(frames):
        cv2.imshow(f"Frame {i}", frame)

        key = cv2.waitKey(0)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

# Takes an array of video frames and creates two separate arrays at time t away from each other.
# t decides how far apart the frames are in time(not necessarily in seconds, depending on the frame rate)
def split_frames(frames, t):
    num_frames = len(frames)

    follower = [] # This camera will be behind
    lead = [] # This camera will be ahead

    for i in range(num_frames - t):
        frame = frames[i]
        if i < t:
            follower.append(frame)
        else:
            lead.append(frame)
            follower.append(frame)

    lead.extend(frames[-t:])
    return follower, lead

# Iterate through the follower and lead frames side by side
def show_split_frames(follower, lead):
    num_follower = len(follower)
    num_lead = len(lead)

    cv2.namedWindow("Lead and Follower Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lead and Follower Camera", (960, 480))

    for i in range(min(num_follower, num_lead)):
        follower_frame = follower[i]
        lead_frame = lead[i]

        combined_frame = cv2.hconcat([follower_frame, lead_frame])
        cv2.imshow("Lead and Follower Camera", combined_frame)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

frames = video_to_frames('./videos/vid1.mp4', './frames', frame_rate=1)
follower, lead = split_frames(frames, 3)
show_split_frames(follower, lead)
