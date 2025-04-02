import numpy as np
import datetime
import cv2
import subprocess
import re
import csv
from ultralytics import YOLO

from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet

from helper import create_video_writer


# define some parameters
conf_threshold = 0.5
max_cosine_distance = 0.4
nn_budget = None

# Initialize the video capture and the video writer objects
video_cap = cv2.VideoCapture("WW1-3.mp4")
writer = create_video_writer(video_cap, "output.mp4")

# Initial Constant Rate Factor for compression quality
crf_value = 41
fps_value = 16
output_width = 720
output_height = 405
object_detected = False
output_file = '/home/nueng/ptech/output-videos/WW1-3.mp4'

# Function to extract video information using FFmpeg
def extract_video_info(video_file):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=r_frame_rate,nb_frames,bit_rate',
           '-of', 'default=noprint_wrappers=1', video_file]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = result.stdout

    fps_match = re.search(r'r_frame_rate=([0-9]+)/([0-9]+)', output)
    frames_match = re.search(r'nb_frames=([0-9]+)', output)
    bitrate_match = re.search(r'bit_rate=([0-9]+)', output)

    fps = eval(fps_match.group(1) + '/' + fps_match.group(2)) if fps_match else None
    frames = int(frames_match.group(1)) if frames_match else None
    bitrate = int(bitrate_match.group(1)) / 1000 if bitrate_match else None  # bitrate in kbps

    return fps, frames, bitrate

# FFmpeg command to compress video frames
ffmpeg_cmd = [
    'ffmpeg',
    '-y',  # Overwrite output file if exists
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-s', f'{output_width}x{output_height}',
    '-pix_fmt', 'bgr24',
    '-r', str(fps_value), #fps_value
    '-i', '-',
    '-c:v', 'libx264',
    '-crf', str(crf_value), #crf_value
    output_file
]

# Start FFmpeg process
ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# Initialize the YOLOv8 model using the default weights
model = YOLO("best.pt")

# Initialize the deep sort tracker
model_filename = "config/mars-small128.pb"
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# load the COCO class labels the YOLO model was trained on
classes_path = "config/coco.names"
with open(classes_path, "r") as f:
    class_names = f.read().strip().split("\n")

# create a list of random colors to represent each class
np.random.seed(42)  # to get the same colors
colors = np.random.randint(0, 255, size=(len(class_names), 3))  # (80, 3)

# loop over the frames
while True:
    # starter time to computer the fps
    start = datetime.datetime.now()
    ret, frame = video_cap.read()

    # if there is no frame, we have reached the end of the video
    if not ret:
        print("End of the video file...")
        break

    ############################################################
    ### Detect the objects in the frame using the YOLO model ###
    ############################################################

    # run the YOLO model on the frame
    results = model(frame)

    # loop over the results
    for result in results:
        # initialize the list of bounding boxes, confidences, and class IDs
        bboxes = []
        confidences = []
        class_ids = []

        # loop over the detections
        for data in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = data
            x = int(x1)
            y = int(y1)
            w = int(x2) - int(x1)
            h = int(y2) - int(y1)
            class_id = int(class_id)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # filter out weak predictions by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > conf_threshold:
                bboxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

        # Check if objects were detected
        if len(bboxes) > 0:
            objects_detected = True
            crf_value = 23
            fps_value = 20
        else:
            objects_detected = False
            crf_value = 41
            fps_value = 16

    ############################################################
    ### Track the objects in the frame using DeepSort        ###
    ############################################################

    # get the names of the detected objects
    names = [class_names[class_id] for class_id in class_ids]

    # get the features of the detected objects
    features = encoder(frame, bboxes)
    # convert the detections to deep sort format
    dets = []
    for bbox, conf, class_name, feature in zip(bboxes, confidences, names, features):
        dets.append(Detection(bbox, conf, class_name, feature))

    # run the tracker on the detections
    tracker.predict()
    tracker.update(dets)

    # loop over the tracked objects
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        # get the bounding box of the object, the name
        # of the object, and the track id
        bbox = track.to_tlbr()
        track_id = track.track_id
        class_name = track.get_class()
        # convert the bounding box to integers
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # get the color associated with the class name
        class_id = class_names.index(class_name)
        color = colors[class_id]
        B, G, R = int(color[0]), int(color[1]), int(color[2])

        # draw the bounding box of the object, the name
        # of the predicted object, and the track id
        text = str(track_id) + " - " + class_name
        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 20),
                      (x1 + len(text) * 12, y1), (B, G, R), -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    ############################################################
    ### Some post-processing to display the results          ###
    ############################################################

    # Resize frame
    frame = cv2.resize(frame, (output_width, output_height))

    # FFmpeg command to compress video frames with updated CRF and FPS
    ffmpeg_cmd[11] = '-r'  #17
    ffmpeg_cmd[12] = str(fps_value)  #18
    ffmpeg_cmd[17] = '-crf' #13
    ffmpeg_cmd[18] = str(crf_value) #14


    # Write frame to FFmpeg process
    ffmpeg_process.stdin.write(frame.tobytes())

    # end time to compute the fps
    end = datetime.datetime.now()


    # calculate the frame per second and draw it on the frame
    #fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {fps_value}" #{1 / (end - start).total_seconds():.2f}
    crf_text = f"CRF: {crf_value}"

    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    cv2.putText(frame, crf_text, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("Output", frame)
    # write the frame to disk
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break


# release the video capture, video writer, and close all windows
video_cap.release()
writer.release()
cv2.destroyAllWindows()

# Extract video information
fps, frame_count, bitrate = extract_video_info(output_file)

# Write information to CSV
csv_filename = "/home/nueng/ptech/output-videos/WW1-30-out.csv"
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['FPS', 'CRF', 'Frame Count', 'Bitrate (kbps)'])
    writer.writerow([fps, crf_value, frame_count, bitrate])
