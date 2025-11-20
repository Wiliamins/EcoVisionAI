# backend/app.py
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import json
import requests
import os
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths (relative to this file)
BASE_DIR = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(BASE_DIR, "train")
MODEL_PATH = os.path.join(TRAIN_DIR, "trashnet_model.h5")
LABELS_PATH = os.path.join(TRAIN_DIR, "class_map.json")

# --- Try to load model & labels (handle missing gracefully)
model: Optional[tf.keras.Model] = None
id_to_label = {}

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    # model stays None; calls will return error
    print(f"Failed to load model: {e}")

try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    id_to_label = {v: k for k, v in class_map.items()}
except Exception as e:
    print(f"Failed to load class_map.json: {e}")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# simple ruPolice detector by histogram intersection (no opencv)
def is_rupolice_pil(image_bytes, threshold=0.92):
    """
    Compare normalized histograms with samples in backend/rupolice/.
    Returns True if any sample histogram intersection > threshold.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((256, 256))
        h = np.array(img.histogram(), dtype=float)
        s = h.sum()
        if s == 0:
            return False
        h /= s

        folder = os.path.join(BASE_DIR, "rupolice")
        if not os.path.isdir(folder):
            return False

        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                sample = Image.open(fpath).convert("RGB").resize((256, 256))
                h2 = np.array(sample.histogram(), dtype=float)
                s2 = h2.sum()
                if s2 == 0:
                    continue
                h2 /= s2
                score = float(np.sum(np.minimum(h, h2)))
                if score > threshold:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False

@app.get("/ping")
def ping():
    return {"success": True, "message": "ok"}

@app.get("/air_quality")
def air_quality(lat: float, lon: float):
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=us_aqi"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        aqi_list = data.get("hourly", {}).get("us_aqi", [])
        if not aqi_list:
            return {"success": False, "error": "AQI data unavailable"}

        # choose latest available (first)
        aqi = aqi_list[0]

        desc = "Good"
        if aqi > 50: desc = "Moderate"
        if aqi > 100: desc = "Unhealthy"
        if aqi > 150: desc = "Very Unhealthy"
        if aqi > 200: desc = "Hazardous"

        return {"success": True, "aqi": int(aqi), "description": desc}

    except requests.RequestException as e:
        return {"success": False, "error": f"Network error: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # read bytes once
    try:
        file_bytes = await file.read()
    except Exception as e:
        return {"success": False, "error": f"Could not read file: {e}"}

    # ruPolice quick-check
    if is_rupolice_pil(file_bytes):
        # return unified "result" structure to frontend
        return {
            "success": True,
            "result": {
                "label": "rupolice",
                "confidence": 1.0,
                "message": "Detected special class: rupolice"
            }
        }

    if model is None:
        return {"success": False, "error": "Model not loaded on server."}
    if not id_to_label:
        return {"success": False, "error": "Labels mapping not found."}

    try:
        tensor = preprocess_image(file_bytes)
        preds = model.predict(tensor)[0]
        top3_ids = preds.argsort()[-3:][::-1]

        top3 = [{"label": id_to_label[int(idx)], "confidence": float(preds[int(idx)])} for idx in top3_ids]

        best_idx = int(np.argmax(preds))
        best_label = id_to_label[best_idx]
        best_conf = float(preds[best_idx])

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
