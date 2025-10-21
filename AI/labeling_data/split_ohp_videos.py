import os
import json
import shutil

# Calea către folderul unde ai video-urile dezarhivate
IMAGES_DIR = '../data_raw/OHP/OHP/Labeled_Dataset/videos/videos'  # sau schimbă dacă ai pus alt nume
OUTPUT_CORRECT = 'data/ohp/correct'
OUTPUT_INCORRECT = 'data/ohp/incorrect'

# Creează folderele de output dacă nu există
os.makedirs(OUTPUT_CORRECT, exist_ok=True)
os.makedirs(OUTPUT_INCORRECT, exist_ok=True)

# Listează fișierele JSON de erori
error_files = [
    'data_raw\OHP\OHP\Labeled_Dataset\Labels\error_knees.json',
    'data_raw\OHP\OHP\Labeled_Dataset\Labels\error_elbows.json'
]

# 1. Citim toate fișierele de erori
incorrect_ids = set()

for error_file in error_files:
    with open(error_file, 'r') as f:
        errors = json.load(f)
        for video_id, error in errors.items():
            if error:  # Dacă există o listă de erori nenulă
                incorrect_ids.add(video_id)

print(f"✅ Total video-uri incorecte detectate: {len(incorrect_ids)}")


# 2. Parcurgem toate video-urile și le sortăm
for video_name in os.listdir(IMAGES_DIR):
    video_path = os.path.join(IMAGES_DIR, video_name)

    # Verificăm dacă este FIȘIER video real
    if not os.path.isfile(video_path):
        continue

    # Extragem ID-ul video-ului fără extensie
    video_id = os.path.splitext(video_name)[0]

    # Decidem destinația
    if video_id in incorrect_ids:
        dest_path = os.path.join(OUTPUT_INCORRECT, video_name)
    else:
        dest_path = os.path.join(OUTPUT_CORRECT, video_name)

    shutil.copy(video_path, dest_path)

print(f"✅ Sortare completă:")
print(f"   - {len(os.listdir(OUTPUT_CORRECT))} video-uri corecte")
print(f"   - {len(os.listdir(OUTPUT_INCORRECT))} video-uri incorecte")