import cv2
import pandas as pd 
import matplotlib.pyplot as plt


df = pd.read_csv('../input/nfl-health-and-safety-helmet-assignment/train_labels.csv')
selected_video_frame_details = {}
video_to_process = cv2.VideoCapture('../input/nfl-health-and-safety-helmet-assignment/train/57583_000082_Endzone.mp4')

fps = int(video_to_process.get(cv2.CAP_PROP_FPS))
width = int(video_to_process.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_to_process.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(video_to_process.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(total_frames):
    selected_video_frame_details[i] = []

for row in range(len(df['video'])):
    if df['video'][row] == '57583_000082_Endzone.mp4':
        try:
            selected_video_frame_details[df['frame'][row]].append([ df['left'][row], df['width'][row], df['top'][row], df['height'][row] ])
        except  KeyError:
            print("index issue has occured")

frames, helmets = [],[]

for i in range(len(selected_video_frame_details)):
    frames.append(i)
    helmets.append(len(selected_video_frame_details[i]))
    # frames_map.append(len(selected_video_frame_details[i]))

plt.bar(frames, helmets, align='center') 
plt.xlabel('Frame')
plt.ylabel('Helmet count')
plt.show()

print("No of frames")
print(total_frames)
