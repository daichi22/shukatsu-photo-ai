import cv2
import numpy as np
import mediapipe as mp

# Mediapipe初期化（グローバルで使う）
mp_face_mesh = mp.solutions.face_mesh

def extract_and_draw_features(image):
    img_bgr = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = img_rgb.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img_rgb)

    annotated_image = img_rgb.copy()

    if not results.multi_face_landmarks:
        return None, None, None, None, annotated_image

    face_landmarks = results.multi_face_landmarks[0]

    # --- 明るさ（顔領域） ---
    x_coords = [landmark.x for landmark in face_landmarks.landmark]
    y_coords = [landmark.y for landmark in face_landmarks.landmark]
    x_min, x_max = int(min(x_coords) * width), int(max(x_coords) * width)
    y_min, y_max = int(min(y_coords) * height), int(max(y_coords) * height)

    face_roi = cv2.cvtColor(img_rgb[y_min:y_max, x_min:x_max], cv2.COLOR_RGB2GRAY)
    brightness = np.mean(face_roi) if face_roi.size > 0 else 0

    # --- 笑顔スコア ---
    left_mouth = face_landmarks.landmark[61]
    right_mouth = face_landmarks.landmark[291]
    left_eye = face_landmarks.landmark[130]
    right_eye = face_landmarks.landmark[359]

    mouth_dist = np.linalg.norm([left_mouth.x - right_mouth.x, left_mouth.y - right_mouth.y])
    eye_dist = np.linalg.norm([left_eye.x - right_eye.x, left_eye.y - right_eye.y])
    smile_score = mouth_dist / eye_dist if eye_dist > 0 else 0

    # --- 顔の傾き ---
    left_eye_y = face_landmarks.landmark[159].y
    right_eye_y = face_landmarks.landmark[386].y
    tilt_score = abs(left_eye_y - right_eye_y)

    # --- 描画処理 ---
    cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return brightness, smile_score, tilt_score, face_landmarks, annotated_image
