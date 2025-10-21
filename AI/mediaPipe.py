# Reimport necesar după resetarea contextului
import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Setări generale
IMAGE_SIZE = (128, 128)
POSE_LANDMARKS = mp.solutions.pose.PoseLandmark

# Inițializare MediaPipe pose
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True)

def extract_landmark_angles(image_path):
    """
    Primește calea către o imagine, returnează un vector de unghiuri între articulații.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    def get_point(name):
        lm = landmarks[name.value]
        return np.array([lm.x, lm.y])

    def compute_angle(a, b, c):
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        return np.degrees(angle)

    angles = []
    triplets = [
        (POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.LEFT_KNEE, POSE_LANDMARKS.LEFT_ANKLE),
        (POSE_LANDMARKS.RIGHT_HIP, POSE_LANDMARKS.RIGHT_KNEE, POSE_LANDMARKS.RIGHT_ANKLE),
        (POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.LEFT_ELBOW, POSE_LANDMARKS.LEFT_WRIST),
        (POSE_LANDMARKS.RIGHT_SHOULDER, POSE_LANDMARKS.RIGHT_ELBOW, POSE_LANDMARKS.RIGHT_WRIST),
    ]

    for a, b, c in triplets:
        try:
            angle = compute_angle(get_point(a), get_point(b), get_point(c))
            angles.append(angle)
        except:
            angles.append(0.0)

    return np.array(angles)

# Dummy: cale exemplu cu imagini corecte
def extract_features_from_folder(folder_path, selected_ids):
    features = []
    valid_ids = []
    for fname in tqdm(os.listdir(folder_path), desc=f"Procesez {folder_path}"):
        img_id = os.path.splitext(fname)[0]
        if img_id in selected_ids:
            path = os.path.join(folder_path, fname)
            angles = extract_landmark_angles(path)
            if angles is not None:
                features.append(angles)
                valid_ids.append(img_id)
    return np.array(features), valid_ids

# Apel pe un folder dummy (te rog să-l adaptezi ulterior)
features, valid_ids = extract_features_from_folder("data_augmented/barbellrow/correct", selected_ids=set())
import pandas as pd
import ace_tools as tools; tools.display_dataframe_to_user(name="Unghiuri extrase din imagini", dataframe=pd.DataFrame(features))
