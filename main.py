from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import io
import os

# Directory for YOLO output
YOLO_OUTPUT_DIR = "./runs/detect"
os.makedirs(YOLO_OUTPUT_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to serve YOLO detection images
app.mount("/runs", StaticFiles(directory="runs"), name="runs")

# yolov5_path = os.path.abspath("./yolov5")

# Get the directory of the current file
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dynamic paths to the YOLOv5 directory and model file
# YOLO_DIR = os.path.join(BASE_DIR, "yolov5")
# MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# model = torch.hub.load(YOLO_DIR, 'custom', path=MODEL_PATH, device='cpu')

# # Load the YOLOv5 model
model = torch.hub.load('./yolov5', 'custom', path='./model/best.pt')

# Load the YOLOv5 model
#model = torch.hub.load('./yolov5', 'custom', path='./model/best.pt', source='local', device='cpu')


@app.get("/")
async def root():
    return {"message": "Welcome to the Pothole Detection API"}


@app.post("/analyze-pothole")
async def analyze_pothole(file: UploadFile):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPEG or PNG image.")

    # Load image from uploaded file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    # Perform pothole detection using YOLOv5
    results = model(img)
    results.save(YOLO_OUTPUT_DIR)

    # Get the latest YOLO output folder
    exp_folders = sorted(
        [d for d in os.listdir(YOLO_OUTPUT_DIR) if os.path.isdir(os.path.join(YOLO_OUTPUT_DIR, d))],
        key=lambda x: os.path.getmtime(os.path.join(YOLO_OUTPUT_DIR, x))
    )
    latest_exp_path = os.path.join(YOLO_OUTPUT_DIR, exp_folders[-1])

    # Assume the labeled image is named "image0.jpg"
    labeled_image_name = "image0.jpg"
    labeled_image_path = os.path.join(latest_exp_path, labeled_image_name)

    if not os.path.exists(labeled_image_path):
        raise HTTPException(status_code=404, detail="Labeled image not found.")

    # Analyze detection results
    num_detections = results.xyxy[0].shape[0]
    severity = "low" if num_detections < 3 else "high"

    return {
        "severity": severity,
        "num_potholes_detected": num_detections,
        "labeled_image_url": f"/runs/detect/{exp_folders[-1]}/{labeled_image_name}"
    }
