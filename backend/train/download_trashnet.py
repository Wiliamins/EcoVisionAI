import os
import zipfile
import requests
from io import BytesIO
import shutil

os.makedirs("data", exist_ok=True)
url = "https://github.com/garythung/trashnet/archive/refs/heads/master.zip"

print("⬇️ Скачиваю TrashNet...")
r = requests.get(url, stream=True)
z = zipfile.ZipFile(BytesIO(r.content))
z.extractall("tmp_trashnet")
print("✅ Распаковано.")

# Обновлённый путь к папке с классами
# Внутри tmp_trashnet/trashnet-master/TrashNet есть папки cardboard, glass, ...
src_root = "tmp_trashnet/trashnet-master/TrashNet"
if not os.path.exists(src_root):
    # если структура другая, ищем
    src_root = "tmp_trashnet/trashnet-master/data"

for d in os.listdir(src_root):
    full = os.path.join(src_root, d)
    if os.path.isdir(full):
        shutil.move(full, os.path.join("data", d))

shutil.rmtree("tmp_trashnet")
print("📁 Данные подготовлены. Проверка:")
print(os.listdir("data"))
