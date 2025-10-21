import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

image_path = "images.jpeg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_pose.Pose(static_image_mode=True) as pose:
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("Niciun landmark detectat.")
    else:
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
        )

        scale_width = 200
        height, width = annotated_image.shape[:2]
        scale_ratio = scale_width / width
        resized_image = cv2.resize(annotated_image, (scale_width, int(height * scale_ratio)))

        cv2.imshow("Pose landmarks", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
