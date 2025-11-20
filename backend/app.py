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
from skimage.metrics import structural_similarity as ssim

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
# AIR QUALITY ENDPOINT
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

# ---------------------------
# RUPOLICE CHECK
# ---------------------------

def is_rupolice(image_path, threshold=0.55):
    """
    Проверяет похожесть фотографии на примеры ruPolice.
    threshold = 0.55 — средняя чувствительность
    """
    try:

        query = Image.open(image_path).convert("L").resize((300, 300))
        query_arr = np.array(query)

        base_dir = "rupolice"
        if not os.path.exists(base_dir):
            return False

        for file in os.listdir(base_dir):
            candidate_path = os.path.join(base_dir, file)
            sample = Image.open(candidate_path).convert("L").resize((300, 300))
            sample_arr = np.array(sample)
            score = ssim(query_arr, sample_arr)
            if score > threshold:
                return True
        return False
    except Exception:
        return False

# ---------------------------
# CLASSIFY ENDPOINT
# ---------------------------

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # читаем файл один раз
    img_bytes = await file.read()
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(img_bytes)

    # проверка rupolice
    if is_rupolice(temp_path):
        os.remove(temp_path)
        return {
            "label": "⚠️ Хуйня позорная",
            "message": "Этого мусора можете выбросить в BIO отходы!",
            "confidence": 1.0
        }

    try:
        tensor = preprocess_image(img_bytes)
        preds = model.predict(tensor)[0]
        top3_ids = preds.argsort()[-3:][::-1]

        top3 = [{"label": id_to_label[idx], "confidence": float(preds[idx])} for idx in top3_ids]

        best_id = int(np.argmax(preds))
        best_label = id_to_label[best_id]
        best_conf = float(preds[best_id])

        os.remove(temp_path)

        return {
            "success": True,
            "result": {
                "label": best_label,
                "confidence": best_conf,
                "top": top3
            }
        }

    except Exception as e:
        os.remove(temp_path)
        return {"success": False, "error": str(e)}

# ---------------------------
# RUN SERVER
# ---------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
