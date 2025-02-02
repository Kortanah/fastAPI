# from typing import Optional

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import torch
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLOv5 model (adjust model path if necessary)
model = torch.hub.load('./yolov5', 'custom', path='./model/best.pt', source='local',device='cpu')

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

    # Analyze detection results
    num_detections = results.xyxy[0].shape[0]
    severity = "low" if num_detections < 3 else "high"
    
    return {
        "severity": severity,
        "num_potholes_detected": num_detections
    }
