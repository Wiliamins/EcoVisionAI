import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import json
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# LOAD MODEL + LABEL MAP

MODEL_PATH = "../backend/train/trashnet_model.h5"
LABELS_PATH = "../backend/train/class_map.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_map = json.load(f)

id_to_label = {v: k for k, v in class_map.items()}

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))      
    img = np.array(img, dtype=np.float32)
    img = img / 255.0                  
    img = np.expand_dims(img, axis=0)
    return img


# ---------------------------
# API ENDPOINT
# ---------------------------

@app.get("/air_quality")
def air_quality(lat: float, lon: float):
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=us_aqi"
        resp = requests.get(url).json()

        aqi = resp["hourly"]["us_aqi"][0]

        desc = "Good"
        if aqi > 50: desc = "Moderate"
        if aqi > 100: desc = "Unhealthy"
        if aqi > 150: desc = "Very Unhealthy"
        if aqi > 200: desc = "Hazardous"

        return {"success": True, "aqi": aqi, "description": desc}

    except Exception as e:
        return {"success": False, "error": str(e)}



@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()

        # 💡 Correct preprocessing
        tensor = preprocess_image(img_bytes)

        # Predict
        preds = model.predict(tensor)[0]     # shape [6]
        top3_ids = preds.argsort()[-3:][::-1]

        top3 = []
        for idx in top3_ids:
            top3.append({
                "label": id_to_label[idx],
                "confidence": float(preds[idx])
            })

        best_id = int(np.argmax(preds))
        best_label = id_to_label[best_id]
        best_conf = float(preds[best_id])

        return {
            "success": True,
            "result": {
                "label": best_label,
                "confidence": best_conf,
                "top": top3
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
