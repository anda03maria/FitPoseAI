import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose

def extract_landmarks_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            return np.array([
                [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
            ]).flatten()
        else:
            return None

def process_folder(folder_path, label):
    features, labels = [], []
    total_images = 0
    successful = 0
    for i, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            total_images += 1
            path = os.path.join(folder_path, filename)
            landmark_vec = extract_landmarks_from_image(path)
            if landmark_vec is not None:
                features.append(landmark_vec)
                labels.append(label)
                successful += 1
            if total_images % 100 == 0:
                print(f"[INFO] Procesate {total_images} imagini... ({successful} cu landmark-uri valide)")

    print(f"[DONE] {successful} din {total_images} imagini procesate cu succes în folderul: {folder_path}")
    return features, labels

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
correct_path = os.path.join(base_dir, 'data_augmented', 'barbellrow', 'correct')
incorrect_path = os.path.join(base_dir, 'data_augmented', 'barbellrow', 'incorrect')


print("FOLDER SCAN:", correct_path)
print("avem?", os.path.exists(correct_path))
print("FOLDER SCAN:", incorrect_path)
print("avem?", os.path.exists(incorrect_path))

features_correct, labels_correct = process_folder(correct_path, 1)
features_incorrect, labels_incorrect = process_folder(incorrect_path, 0)

X = np.array(features_correct + features_incorrect)
y = np.array(labels_correct + labels_incorrect)

np.save('X_landmarks.npy', X)
np.save('y_labels.npy', y)
