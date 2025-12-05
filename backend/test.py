# process video in batches
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch
from tqdm import tqdm

# Configuration
MODEL_REPO = "Subh775/Threat-Detection-YOLOv8n"
INPUT_VIDEO = "input_video.mp4"
OUTPUT_VIDEO = "output_video.mp4"
CONFIDENCE_THRESHOLD = 0.4
BATCH_SIZE = 32  # Adjust based on GPU memory

# Setup device
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load model
model_path = hf_hub_download(repo_id=MODEL_REPO, filename="weights/best.pt")
model = YOLO(model_path)

# Process video
cap = cv2.VideoCapture(INPUT_VIDEO)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

frames_batch = []
with tqdm(total=total_frames, desc="Processing video") as pbar:
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frames_batch.append(frame)
            
            if len(frames_batch) == BATCH_SIZE:
                # Batch inference
                results = model(frames_batch, conf=CONFIDENCE_THRESHOLD, 
                              device=device, verbose=False)
                
                # Write annotated frames
                for result in results:
                    annotated_frame = result.plot()
                    out.write(annotated_frame)
                
                pbar.update(len(frames_batch))
                frames_batch = []
        else:
            break

# Process remaining frames
if frames_batch:
    results = model(frames_batch, conf=CONFIDENCE_THRESHOLD, 
                   device=device, verbose=False)
    for result in results:
        annotated_frame = result.plot()
        out.write(annotated_frame)
    pbar.update(len(frames_batch))

cap.release()
out.release()
print(f"Processed video saved to: {OUTPUT_VIDEO}")
