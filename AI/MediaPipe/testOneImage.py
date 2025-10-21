import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from sklearn.preprocessing import StandardScaler

# Încarcă modelul
model = tf.keras.models.load_model('landmark_model.h5')

# Încarcă scalerul folosit
X = np.load('X_landmarks.npy')  # presupune că ai folosit acest set pentru scaler
scaler = StandardScaler()
scaler.fit(X)  # fit pe datele originale

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Imaginea nu a putut fi citită.")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        return landmarks
    return None

# === Testare pe o imagine ===
image_path = '"C:\\Users\\Anda\\OneDrive\\Imagini\\desktop-pendlay-row.jpg"'  # <-- modifică aici

landmarks = extract_landmarks(image_path)
if landmarks is not None:
    landmarks_scaled = scaler.transform([landmarks])
    prediction = model.predict(landmarks_scaled)
    label = 'Correct' if prediction > 0.5 else 'Incorrect'
    print(f'Predicted posture: {label} (confidence: {prediction[0][0]:.2f})')
else:
    print("Nu s-au detectat landmark-uri.")
