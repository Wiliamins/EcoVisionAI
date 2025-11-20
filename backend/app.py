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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# LOAD MODEL + LABEL MAP
# ---------------------------

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
# SIMPLE RUPOLICE DETECTOR (без OpenCV)
# ---------------------------

def is_rupolice_pil(image_bytes, threshold=0.92):
    """
    Определение ruPolice по похожести гистограмм (без cv2).
    Работает на любых серверах, даже Render.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((256, 256))
        hist = np.array(img.histogram(), dtype=float)
        hist /= hist.sum()

        folder = "rupolice"
        if not os.path.exists(folder):
            return False

        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            try:
                sample = Image.open(path).convert("RGB").resize((256, 256))
                hist2 = np.array(sample.histogram(), dtype=float)
                hist2 /= hist2.sum()

                score = np.sum(np.minimum(hist, hist2))  # пересечение гистограмм

                if score > threshold:
                    return True

            except:
                continue

        return False

    except:
        return False


# ---------------------------
# API
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

    file_bytes = await file.read()  # читаем 1 раз

    # ---- ruPolice check ----
    if is_rupolice_pil(file_bytes):
        return {
            "label": "⚠️ Хуйня позорная",
            "message": "Этого мусора можете выбросить в BIO отходы!",
            "confidence": 1.0
        }

    try:
        tensor = preprocess_image(file_bytes)

        preds = model.predict(tensor)[0]  # shape [6]
        top3_ids = preds.argsort()[-3:][::-1]

        top3 = [{
            "label": id_to_label[idx],
            "confidence": float(preds[idx])
        } for idx in top3_ids]

        best_id = int(np.argmax(preds))
        best_label = id_to_label[best_id]

        return {
            "success": True,
            "result": {
                "label": best_label,
                "confidence": float(preds[best_id]),
                "top": top3
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
