import cv2
import os
import mediapipe as mp
import numpy as np
import csv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_features_extended(landmarks):
    idx = mp_pose.PoseLandmark
    def pt(p): return [landmarks[p.value].x, landmarks[p.value].y]

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
    back_angle_L   = calc_angle(shoulder_L, hip_L, knee_L)
    knee_angle_L   = calc_angle(hip_L, knee_L, ankle_L)
    arm_angle_L    = calc_angle(shoulder_L, elbow_L, wrist_L)
    trunk_angle_L  = calc_angle(hip_L, shoulder_L, [shoulder_L[0], 0])  # față de verticală
    torso_bend_L   = calc_angle(elbow_L, shoulder_L, hip_L)

    # Unghiuri dreapta
    back_angle_R   = calc_angle(shoulder_R, hip_R, knee_R)
    knee_angle_R   = calc_angle(hip_R, knee_R, ankle_R)
    arm_angle_R    = calc_angle(shoulder_R, elbow_R, wrist_R)
    trunk_angle_R  = calc_angle(hip_R, shoulder_R, [shoulder_R[0], 0])
    torso_bend_R   = calc_angle(elbow_R, shoulder_R, hip_R)

    # Opțional: adaugă unghiul dintre cei doi umeri și șold (pentru alinierea trunchiului)
    # midline_tilt = calc_angle(shoulder_L, hip_L, shoulder_R)

    return [
        back_angle_L, knee_angle_L, arm_angle_L, trunk_angle_L, torso_bend_L,
        back_angle_R, knee_angle_R, arm_angle_R, trunk_angle_R, torso_bend_R
        # , midline_tilt  ← dacă vrei să adaugi și această caracteristică
    ]

# Căi
folders = {'correct': 1, 'incorrect': 0}
output_csv = 'barbellrow_dataset.csv'

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'image_id',
        'back_angle_L', 'knee_angle_L', 'arm_angle_L', 'trunk_angle_L', 'torso_bend_L',
        'back_angle_R', 'knee_angle_R', 'arm_angle_R', 'trunk_angle_R', 'torso_bend_R',
        'label'
    ])

    for folder, label in folders.items():
        folder_path = f'../data_augmented/barbellrow/{folder}'
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                try:
                    features = extract_features_extended(results.pose_landmarks.landmark)
                    writer.writerow([filename] + features + [label])
                    print(f'[OK] {filename}')
                except Exception as e:
                    print(f'[SKIP] {filename} - error: {e}')
            else:
                print(f'[NO POSE] {filename}')


