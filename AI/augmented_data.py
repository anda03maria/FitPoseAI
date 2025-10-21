import os
import random
from PIL import Image, ImageEnhance
from torchvision import transforms
from tqdm import tqdm

correct_dir = 'data/barbellrow/correct'
incorrect_dir = 'data/barbellrow/incorrect'
output_correct = 'data_augmented/barbellrow/correct'
output_incorrect = 'data_augmented/barbellrow/incorrect'

os.makedirs(output_correct, exist_ok=True)
os.makedirs(output_incorrect, exist_ok=True)

# Subsample pentru clasa „correct” (10000 imagini)
all_correct = os.listdir(correct_dir)
selected_correct = random.sample(all_correct, 10000)
for fname in tqdm(selected_correct, desc="Copying correct samples"):
    img = Image.open(os.path.join(correct_dir, fname))
    img.save(os.path.join(output_correct, fname))

# Augmentari posibile
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0))
])

all_incorrect = os.listdir(incorrect_dir)
images_needed = 10000
n_original = len(all_incorrect)
n_to_generate = images_needed - n_original

for fname in tqdm(all_incorrect, desc="Copying original incorrect"):
    img = Image.open(os.path.join(incorrect_dir, fname))
    img.save(os.path.join(output_incorrect, fname))

augment_index = 0
while augment_index < n_to_generate:
    fname = random.choice(all_incorrect)
    img = Image.open(os.path.join(incorrect_dir, fname))
    img = img.convert("RGB")
    aug_img = augmentations(img)
    aug_fname = f"aug_{augment_index}_{fname}"
    aug_img.save(os.path.join(output_incorrect, aug_fname))
    augment_index += 1

print("Set echilibrat generat în `data_augmented/barbellrow`")
