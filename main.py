from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import cv2
import xgboost as xgb
import json

app = FastAPI()

# Load pre-trained models
crop_monitor_model = tf.keras.models.load_model("plant_disease_fast.h5")
# For pest detection, integrate YOLOv8 inference (e.g., using the ultralytics package)
pest_detection_model = "yolo_model_placeholder"  # Replace with actual model loading/inference code
resource_optimizer = xgb.Booster()
resource_optimizer.load_model("resource_xgboost.json")

# Define a Pydantic model for resource optimization input
class ResourceInput(BaseModel):
    Crop: str
    Crop_Year: int
    Season: str
    State: str
    Area: float
    Production: float
    Annual_Rainfall: float
    Fertilizer: float
    Pesticide: float

@app.post("/monitor-crop")
async def monitor_crop(file: UploadFile = File(...)):
    # Read and preprocess image
    img_bytes = await file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = crop_monitor_model.predict(img)
    # Map prediction to class label (update as needed)
    class_labels = ["Healthy", "DiseaseA", "DiseaseB"]  # Example labels
    result = class_labels[np.argmax(prediction)]
    return {"crop_status": result}

@app.post("/detect-pest")
async def detect_pest(file: UploadFile = File(...)):
    # Placeholder: Integrate YOLOv8 inference here
    # Process the image, run YOLO model, and return detected pests
    return {"pest_status": "Pest detected: aphids", "details": {"confidence": 0.85}}

@app.post("/optimize-resources")
async def optimize_resources(data: ResourceInput):
    # Convert input data to model features (ensure proper encoding/preprocessing)
    features = np.array([[data.Crop, data.Crop_Year, data.Season, data.State, data.Area, data.Production, data.Annual_Rainfall, data.Fertilizer, data.Pesticide]])
    dmatrix = xgb.DMatrix(features)
    yield_pred = resource_optimizer.predict(dmatrix)[0]
    return {"predicted_yield": yield_pred}

@app.post("/recommend")
async def recommend(crop_file: UploadFile = File(...),
                    pest_file: UploadFile = File(...),
                    crop_year: int = Form(...),
                    season: str = Form(...),
                    state: str = Form(...),
                    area: float = Form(...),
                    production: float = Form(...),
                    annual_rainfall: float = Form(...),
                    fertilizer: float = Form(...),
                    pesticide: float = Form(...)):
    # Get crop monitoring result
    crop_result = await monitor_crop(crop_file)
    # Get pest detection result
    pest_result = await detect_pest(pest_file)
    # Get resource optimization result
    resource_input = ResourceInput(
        Crop=crop_result.get("crop_status"),  # Optionally use crop monitoring result here
        Crop_Year=crop_year,
        Season=season,
        State=state,
        Area=area,
        Production=production,
        Annual_Rainfall=annual_rainfall,
        Fertilizer=fertilizer,
        Pesticide=pesticide
    )
    resource_result = await optimize_resources(resource_input)
    
    # Business Logic for Recommendation (simplified example)
    recommendations = (
        f"Crop Health: {crop_result['crop_status']}. "
        f"Pest Status: {pest_result['pest_status']}. "
        f"Predicted Yield: {resource_result['predicted_yield']:.2f}. "
        "Based on these results, we recommend targeted interventions for pest control and adjusting water/fertilizer inputs."
    )
    
    return {"final_recommendation": recommendations}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
