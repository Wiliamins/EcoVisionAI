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
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

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

def is_rupolice(image_path, threshold=0.55):
    """
    Проверяет похожесть фотографии на примеры ruPolice.
    threshold = 0.55 — средняя чувствительность
    """
    try:
        query = cv2.imread(image_path)
        if query is None:
            return False

        query = cv2.resize(query, (300, 300))
        query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

        base_dir = "rupolice"
        if not os.path.exists(base_dir):
            return False

        for file in os.listdir(base_dir):
            candidate_path = os.path.join(base_dir, file)
            sample = cv2.imread(candidate_path)
            if sample is None:
                continue

            sample = cv2.resize(sample, (300, 300))
            sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

            # сравнение
            score = ssim(query_gray, sample_gray)

            if score > threshold:
                return True

        return False

    except Exception:
        return False


@app.post("/classify")
async def classify(file: UploadFile = File(...)):

    temp_path = f"temp_{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    if is_rupolice(temp_path):
        os.remove(temp_path)
        return {
            "label": "⚠️ Хуйня позорная",
            "message": "Этого мусора можете выбросить в BIO отходы!",
            "confidence": 1.0
        }

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
