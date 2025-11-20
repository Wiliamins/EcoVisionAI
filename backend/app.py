from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

RUPOLICE_KEYWORDS = ["rupolice", "rupolice.ru", "ruspolice", "полиц", "дпс", "мвд"]

def is_rupolice_image(image_bytes: bytes) -> bool:
    filename = ""  # нет имени — просто проверяем содержимое
    content_lower = image_bytes[:2000].decode('latin-1', errors='ignore').lower()
    
    return any(word in content_lower for word in RUPOLICE_KEYWORDS)

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # 🔥 Проверка rupolice
        if is_rupolice_image(image_bytes):
            return {
                "label": "error_rupolice",
                "confidence": 1.0,
                "message": "Хуйня позорная. Фото не принимается."
            }

        # Простейшая «классификация»
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except:
            return {
                "label": "invalid_image",
                "confidence": 0.0,
                "message": "Ошибка: изображение повреждено."
            }

        # ДЕМО-логика
        np_img = np.array(image)
        brightness = np.mean(np_img)

        if brightness < 60:
            label = "dark_object"
        elif brightness > 200:
            label = "light_object"
        else:
            label = "medium_object"

        return {
            "label": label,
            "confidence": float(round(brightness / 255, 2)),
            "message": "OK"
        }

    except Exception as e:
        return {
            "label": "backend_error",
            "confidence": 0.0,
            "message": f"Error: {str(e)}"
        }
