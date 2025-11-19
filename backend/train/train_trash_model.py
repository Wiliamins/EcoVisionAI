import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
import json
import os

# Путь к данным
data_dir = "../train/data"

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Папка {data_dir} не найдена!")

# Аугментации + автоматический split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Сохраняем mapping классов
with open("class_map.json", "w") as f:
    json.dump(train_gen.class_indices, f)
print("Saved class_map.json")
print("Classes:", train_gen.class_indices)

# Базовая модель
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Глобальный пуллинг + Dense слои
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
preds = Dense(len(train_gen.class_indices), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=preds)

# 🔥 Важное: разморозим последние 25 слоёв для обучения нового класса
for layer in base_model.layers[:-25]:
    layer.trainable = False
for layer in base_model.layers[-25:]:
    layer.trainable = True

# Компиляция модели
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Обучение
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    verbose=1
)

# Сохраняем модель
model.save("trashnet_model.h5")
print("Модель успешно сохранена!")
