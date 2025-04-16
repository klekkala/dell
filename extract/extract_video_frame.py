import cv2
import os
import sys

video_path = sys.argv[1]
video_name = os.path.splitext(os.path.basename(video_path))[0]

output_folder = f"./data/{video_name}"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.png")

#     cv2.imwrite(frame_filename, frame)

#     frame_count += 1

saved_frame_count = 0

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_seconds = total_frames / fps 
max_duration_seconds = 30 * 60
max_frames = int(fps * max_duration_seconds)

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # every 5 frame a picture
    if frame_count % 5 == 0:
        frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:06d}.png")
        
        cv2.imwrite(frame_filename, frame)
        
        saved_frame_count += 1

    frame_count += 1

cap.release()

print(f"Finished, got {frame_count} frame, save to: {output_folder}")