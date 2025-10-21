import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import joblib
import tensorflow as tf
import os
import mediapipe as mp

mp_pose = mp.solutions.pose

from keras.src.saving import load_model

from predict import predict_image_posture_h5, predict_image_posture, predict_video_posture

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Înlocuiește cu frontend-ul tău în producție
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cnn_model = tf.keras.models.load_model("barbellrow.h5", compile=False)
xgb_model, xgb_encoder = joblib.load("model_clasificare_xgb.joblib")
squat_model = tf.keras.models.load_model("squat.h5", compile=False)
ohp_model = tf.keras.models.load_model("overhead_press.h5", compile=False)
br_xgboost_model, classes = joblib.load("model_clasificare_xgb.joblib")

def preprocess_video(video_path, frame_count=20, img_size=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_idx = np.linspace(0, total_frames-1, frame_count, dtype=int)
    frames = []
    for idx in frames_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    # Normalize and add batch dimension
    frames = np.array(frames) / 255.0
    if frames.shape[0] < frame_count:  # pad if too short
        pad = np.zeros((frame_count - frames.shape[0], *img_size, 3))
        frames = np.concatenate([frames, pad])
    frames = np.expand_dims(frames, axis=0)  # shape: (1, time_steps, h, w, 3)
    return frames

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Imaginea nu poate fi încărcată!")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 128, 128, 3)
    return img

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_landmarks_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Imagine invalidă!")
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image)
        if not results.pose_landmarks:
            return None  # Sau np.zeros(10)
        landmarks = results.pose_landmarks.landmark

        def pt(p): return [landmarks[p.value].x, landmarks[p.value].y]

        idx = mp_pose.PoseLandmark

        # Partea stângă
        shoulder_L = pt(idx.LEFT_SHOULDER)
        hip_L = pt(idx.LEFT_HIP)
        knee_L = pt(idx.LEFT_KNEE)
        ankle_L = pt(idx.LEFT_ANKLE)
        elbow_L = pt(idx.LEFT_ELBOW)
        wrist_L = pt(idx.LEFT_WRIST)

        # Partea dreaptă
        shoulder_R = pt(idx.RIGHT_SHOULDER)
        hip_R = pt(idx.RIGHT_HIP)
        knee_R = pt(idx.RIGHT_KNEE)
        ankle_R = pt(idx.RIGHT_ANKLE)
        elbow_R = pt(idx.RIGHT_ELBOW)
        wrist_R = pt(idx.RIGHT_WRIST)

        # Unghiuri stânga
        back_angle_L = calc_angle(shoulder_L, hip_L, knee_L)
        knee_angle_L = calc_angle(hip_L, knee_L, ankle_L)
        arm_angle_L = calc_angle(shoulder_L, elbow_L, wrist_L)
        trunk_angle_L = calc_angle(hip_L, shoulder_L, [shoulder_L[0], 0])
        torso_bend_L = calc_angle(elbow_L, shoulder_L, hip_L)

        # Unghiuri dreapta
        back_angle_R = calc_angle(shoulder_R, hip_R, knee_R)
        knee_angle_R = calc_angle(hip_R, knee_R, ankle_R)
        arm_angle_R = calc_angle(shoulder_R, elbow_R, wrist_R)
        trunk_angle_R = calc_angle(hip_R, shoulder_R, [shoulder_R[0], 0])
        torso_bend_R = calc_angle(elbow_R, shoulder_R, hip_R)

        return [
            back_angle_L, knee_angle_L, arm_angle_L, trunk_angle_L, torso_bend_L,
            back_angle_R, knee_angle_R, arm_angle_R, trunk_angle_R, torso_bend_R
        ]

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(None),
    exercise: str = Form(...)
):
    contents = await file.read()
    temp_file = f"{file.filename}"

    with open(temp_file, "wb") as f:
        f.write(contents)

    if exercise == "Barbell row":
        if model_type == "image":
            image = preprocess_image(temp_file);
            y_pred = cnn_model.predict(image)
            label = "correct" if y_pred[0][0] >= 0.5 else "incorrect"
            score = float(y_pred[0][0])
        elif model_type == "landmark":
            features = extract_landmarks_from_image(temp_file)
            if features is None:
                return 0, 0.0
            features = np.array(features).reshape(1, -1)
            features = np.concatenate([np.zeros((features.shape[0], 1)), features], axis=1)  # Adaugă 0 ca prim feature
            pred = br_xgboost_model.predict(features)
            score = float(br_xgboost_model.predict_proba(features).max())
            if score >= 0.8:
                label = "correct"
            elif 0.5 <= score < 0.8:
                label = "mediu"
            else:
                label = "incorrect"

    elif exercise == "Overhead press":
        frames = preprocess_video(temp_file, frame_count=20, img_size=(64, 64))
        y_pred = ohp_model.predict(frames)
        label = "correct" if y_pred[0][0] >= 0.5 else "incorrect"
        score = float(y_pred[0][0])

    elif exercise == "Squat":
        frames = preprocess_video(temp_file, frame_count=20, img_size=(64, 64))
        y_pred = squat_model.predict(frames)
        label = "correct" if y_pred[0][0] >= 0.5 else "incorrect"
        score = float(y_pred[0][0])

    else:
        return JSONResponse(content={"error": "Unknown exercise"}, status_code=400)

    return {
        "exercise": exercise,
        "result": label,
        "score": score
    }