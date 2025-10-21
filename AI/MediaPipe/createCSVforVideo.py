import os
import cv2
import numpy as np
import mediapipe as mp
import csv

VIDEO_DIR = "C:/Users/Anda/OneDrive/Desktop/LICENTA/PythonProject/data/ohp"
OUTPUT_CSV = "ohp_dataset.csv"

mp_pose = mp.solutions.pose

def get_point(landmarks, index):
    return np.array([landmarks[index].x, landmarks[index].y])

def angle_between(a, b, c):
    ab = a - b
    cb = c - b
    cosine = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_features(frame, pose):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark
    try:
        features = [
            angle_between(get_point(lm, 11), get_point(lm, 23), get_point(lm, 25)),  # back_angle_L
            angle_between(get_point(lm, 23), get_point(lm, 25), get_point(lm, 27)),  # knee_angle_L
            angle_between(get_point(lm, 13), get_point(lm, 11), get_point(lm, 23)),  # arm_angle_L
            angle_between(get_point(lm, 11), get_point(lm, 23), get_point(lm, 24)),  # trunk_angle_L
            angle_between(get_point(lm, 23), get_point(lm, 11), get_point(lm, 13)),  # torso_bend_L

            angle_between(get_point(lm, 12), get_point(lm, 24), get_point(lm, 26)),  # back_angle_R
            angle_between(get_point(lm, 24), get_point(lm, 26), get_point(lm, 28)),  # knee_angle_R
            angle_between(get_point(lm, 14), get_point(lm, 12), get_point(lm, 24)),  # arm_angle_R
            angle_between(get_point(lm, 12), get_point(lm, 24), get_point(lm, 23)),  # trunk_angle_R
            angle_between(get_point(lm, 24), get_point(lm, 12), get_point(lm, 14)),  # torso_bend_R
        ]
        return features
    except:
        return None

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    features_list = []
    with mp_pose.Pose(static_image_mode=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            features = extract_features(frame, pose)
            if features:
                features_list.append(features)
    cap.release()
    if features_list:
        return np.mean(features_list, axis=0)  # vector mediu
    return None

def main():
    rows = []
    for label_dir, label in [("corect", 1), ("incorect", 0)]:
        full_path = os.path.join(VIDEO_DIR, label_dir)
        for fname in os.listdir(full_path):
            if fname.endswith(".mp4"):
                video_path = os.path.join(full_path, fname)
                print(f"Procesare: {video_path}")
                features = process_video(video_path)
                if features is not None:
                    rows.append(list(features) + [label])

    # Scrie în CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            'back_angle_L', 'knee_angle_L', 'arm_angle_L', 'trunk_angle_L', 'torso_bend_L',
            'back_angle_R', 'knee_angle_R', 'arm_angle_R', 'trunk_angle_R', 'torso_bend_R',
            'label'
        ]
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n✅ Salvare completă în: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
