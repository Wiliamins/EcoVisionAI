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
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../train/trashnet_model.h5")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "../train/class_map.json")
RUPOLICE_DIR = os.path.join(os.path.dirname(__file__), "rupolice")  # папка с примерами

# ---------------------------
# LOAD MODEL + LABEL MAP (лениво — логируем ошибки)
# ---------------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    id_to_label = {v: k for k, v in class_map.items()}
    logging.info(f"Class map loaded: {class_map}")
except Exception as e:
    logging.error(f"Failed to load class_map.json: {e}")
    class_map = {}
    id_to_label = {}

# ---------------------------
# UTIL: safe image open
# ---------------------------
def safe_open_image_bytes(image_bytes):
    """Возвращает PIL.Image RGB или None если не получилось открыть"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # проверка целостности
        img = Image.open(io.BytesIO(image_bytes))  # reopen after verify
        return img.convert("RGB")
    except Exception as e:
        logging.warning(f"safe_open_image_bytes: cannot open image: {e}")
        return None

def preprocess_image_bytes(image_bytes):
    img = safe_open_image_bytes(image_bytes)
    if img is None:
        raise ValueError("Invalid or unreadable image")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------
# SIMPLE ruPolice detector (histogram similarity)
# ---------------------------
def is_rupolice_by_hist(image_bytes, threshold=0.92):
    """
    Сравнивает гистограмму загруженного изображения с примерами в папке RUPOLICE_DIR.
    Возвращает True если совпадение больше threshold.
    """
    img = safe_open_image_bytes(image_bytes)
    if img is None:
        return False

    try:
        img = img.resize((256, 256))
        hist = np.array(img.histogram(), dtype=float)
        s = hist.sum()
        if s == 0:
            return False
        hist /= s

        if not os.path.isdir(RUPOLICE_DIR):
            return False

        for fname in os.listdir(RUPOLICE_DIR):
            path = os.path.join(RUPOLICE_DIR, fname)
            try:
                sample = Image.open(path).convert("RGB").resize((256, 256))
                hist2 = np.array(sample.histogram(), dtype=float)
                s2 = hist2.sum()
                if s2 == 0:
                    continue
                hist2 /= s2
                score = float(np.sum(np.minimum(hist, hist2)))
                if score > threshold:
                    logging.info(f"rupolice match: {fname} score={score:.3f}")
                    return True
            except Exception as e:
                logging.debug(f"Skipping sample {path}: {e}")
        return False
    except Exception as e:
        logging.warning(f"is_rupolice_by_hist failed: {e}")
        return False

# ---------------------------
# AIR QUALITY endpoint
# ---------------------------
@app.get("/air_quality")
def air_quality(lat: float, lon: float):
    logging.info(f"air_quality request lat={lat} lon={lon}")
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=us_aqi"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        j = resp.json()
        aqi = j.get("hourly", {}).get("us_aqi")
        if not aqi:
            return {"success": False, "error": "No AQI data from provider"}
        aqi0 = aqi[0]
        desc = "Good"
        if aqi0 > 200: desc = "Hazardous"
        elif aqi0 > 150: desc = "Very Unhealthy"
        elif aqi0 > 100: desc = "Unhealthy"
        elif aqi0 > 50: desc = "Moderate"
        return {"success": True, "aqi": int(aqi0), "description": desc}
    except Exception as e:
        logging.error(f"air_quality error: {e}")
        return {"success": False, "error": str(e)}

# ---------------------------
# CLASSIFY endpoint
# ---------------------------
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    logging.info(f"classify called, filename={file.filename}, content_type={file.content_type}")
    try:
        body = await file.read()
        if not body:
            return {"success": False, "error": "Empty file"}

        # 1) ruPolice check (fast)
        if is_rupolice_by_hist(body):
            # возвращаем согласованный JSON (success + result)
            return {
                "success": True,
                "result": {
                    "label": "rupolice",
                    "display_label": "⚠️ Хуйня позорная",
                    "confidence": 1.0,
                    "message": "Этого мусора можете выбросить в БИО отходы"
                }
            }

        # 2) model inference
        if model is None:
            return {"success": False, "error": "Model not loaded on server"}

        tensor = preprocess_image_bytes(body)
        preds = model.predict(tensor)[0]
        # top3
        top3_ids = preds.argsort()[-3:][::-1]
        top3 = [{"label": id_to_label.get(int(i), str(i)), "confidence": float(preds[int(i)])} for i in top3_ids]
        best_id = int(np.argmax(preds))
        best_label = id_to_label.get(best_id, str(best_id))
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
        logging.exception("classify failed")
        return {"success": False, "error": str(e)}

# ---------------------------
if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
