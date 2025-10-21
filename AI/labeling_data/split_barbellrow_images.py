import os
import json
import shutil

# Căile către foldere
IMAGES_DIR = '../data_raw/BarbellRow/BarbellRow/Labeled_Dataset/barbellrow_images_raw/barbellrow_images_raw'  # Folderul unde ai imaginile brute
OUTPUT_CORRECT = 'data/barbellrow/correct'
OUTPUT_INCORRECT = 'data/barbellrow/incorrect'

# Creează folderele de output
os.makedirs(OUTPUT_CORRECT, exist_ok=True)
os.makedirs(OUTPUT_INCORRECT, exist_ok=True)

# Fișierele JSON cu etichete de erori
error_files = [
    'data_raw\BarbellRow\BarbellRow\Labeled_Dataset\Labels\labels_lumbar_error.json',
    'data_raw\BarbellRow\BarbellRow\Labeled_Dataset\Labels\labels_torso_angle_error.json'
]

# Set de ID-uri de imagini cu erori
incorrect_ids = set()

# 1. Citim toate fișierele de erori
for error_file in error_files:
    with open(error_file, 'r') as f:
        errors = json.load(f)
        for img_id, error in errors.items():
            if error:  # Dacă lista de erori nu este goală
                incorrect_ids.add(img_id)

print(f"✅ Total imagini incorecte detectate: {len(incorrect_ids)}")

# 2. Parcurgem imaginile și le sortăm
for img_name in os.listdir(IMAGES_DIR):
    img_path = os.path.join(IMAGES_DIR, img_name)

    img_id = os.path.splitext(img_name)[0]

    if img_id in incorrect_ids:
        dest_path = os.path.join(OUTPUT_INCORRECT, img_name)
    else:
        dest_path = os.path.join(OUTPUT_CORRECT, img_name)

    shutil.copy(img_path, dest_path)

print(f"✅ Sortare completă:")
print(f"   - {len(os.listdir(OUTPUT_CORRECT))} imagini corecte")
print(f"   - {len(os.listdir(OUTPUT_INCORRECT))} imagini incorecte")
