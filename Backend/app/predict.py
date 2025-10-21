import joblib
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import os

from PIL.Image import Image
from tensorflow.python.keras.models import load_model

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def calc_angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def calc_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (128, 128))
#     image = image / 255.0
#     return np.expand_dims(image, axis=0)

# def extract_pose_features(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"[EROARE] cv2.imread() a returnat None pentru {image_path}")
#     else:
#         print("Ajunge aici")
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image_rgb)
#
#     if not results.pose_landmarks:
#         raise ValueError("No landmarks detected.")
#
#     lm = results.pose_landmarks.landmark
#     idx = mp_pose.PoseLandmark
#     def pt(p): return [lm[p.value].x, lm[p.value].y]
#
#     features = [
#         calc_angle(pt(idx.LEFT_SHOULDER), pt(idx.LEFT_HIP), pt(idx.LEFT_KNEE)),
#         calc_angle(pt(idx.LEFT_HIP), pt(idx.LEFT_KNEE), pt(idx.LEFT_ANKLE)),
#         calc_angle(pt(idx.LEFT_SHOULDER), pt(idx.LEFT_ELBOW), pt(idx.LEFT_WRIST)),
#         calc_angle(pt(idx.LEFT_HIP), pt(idx.LEFT_SHOULDER), [pt(idx.LEFT_SHOULDER)[0], 0]),
#         calc_angle(pt(idx.LEFT_ELBOW), pt(idx.LEFT_SHOULDER), pt(idx.LEFT_HIP)),
#
#         calc_angle(pt(idx.RIGHT_SHOULDER), pt(idx.RIGHT_HIP), pt(idx.RIGHT_KNEE)),
#         calc_angle(pt(idx.RIGHT_HIP), pt(idx.RIGHT_KNEE), pt(idx.RIGHT_ANKLE)),
#         calc_angle(pt(idx.RIGHT_SHOULDER), pt(idx.RIGHT_ELBOW), pt(idx.RIGHT_WRIST)),
#         calc_angle(pt(idx.RIGHT_HIP), pt(idx.RIGHT_SHOULDER), [pt(idx.RIGHT_SHOULDER)[0], 0]),
#         calc_angle(pt(idx.RIGHT_ELBOW), pt(idx.RIGHT_SHOULDER), pt(idx.RIGHT_HIP)),
#
#         # calc_distance(pt(idx.LEFT_SHOULDER), pt(idx.LEFT_HIP)),
#         # calc_distance(pt(idx.LEFT_HIP), pt(idx.LEFT_ANKLE)),
#         # calc_distance(pt(idx.LEFT_ELBOW), pt(idx.LEFT_HIP)),
#     ]
#
#     return np.array(features).reshape(1, -1)

def extract_pose_features(landmarks):
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y])
    return features

def extract_video_frames(video_path, frame_count=20, img_size=(128,128)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // frame_count)
    frames = []
    count = 0
    while len(frames) < frame_count and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.resize(frame, (64, 64))
            frame = frame / 255.0
            frames.append(frame)
        count += 1
    cap.release()
    while len(frames) < frame_count:
        frames.append(np.zeros((64, 64, 3)))
    return np.expand_dims(np.array(frames), axis=0)

xgb_model, xgb_encoder = joblib.load("model_clasificare_xgb.joblib")

# def preprocess_image_h5(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"[EROARE] Imagine invalidă: {image_path}")
#
#     image = cv2.resize(image, (255, 255))  # dimensiunea exactă cerută de modelul .h5
#     image = image / 255.0  # normalizare
#     return np.expand_dims(image, axis=0)

def predict_image_posture_h5(image_path, model_path):
    model = load_model(model_path)
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))
    array = np.array(image) / 255.0
    array = np.expand_dims(array, axis=0)
    prediction = model.predict(array)[0]
    confidence = float(prediction[0])
    label = "corect" if confidence > 0.5 else "incorect"
    return label, confidence

def predict_image_posture(image_path):
    model = joblib.load("model_clasificare_xgb.joblib")
    mp_pose = mp.solutions.pose
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return "fără landmarkuri", 0.0

        features = extract_pose_features(results.pose_landmarks.landmark)
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features])[0].max()
        return prediction, float(confidence)

def predict_video_posture(video_path, model_path, num_frames=20):
    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = []
    current_index = 0
    frame_id = 0

    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id == indices[current_index]:
            frame = cv2.resize(frame, (128, 128))
            frame = frame / 255.0
            frames.append(frame)
            current_index += 1
            if current_index >= len(indices):
                break

        frame_id += 1

    cap.release()

    while len(frames) < num_frames:
        frames.append(np.zeros((128, 128, 3)))

    input_data = np.array(frames)
    input_data = np.expand_dims(input_data, axis=0)

    prediction = model.predict(input_data)[0]
    confidence = float(prediction[0])
    label = "corect" if confidence > 0.5 else "incorect"
    return label, confidence