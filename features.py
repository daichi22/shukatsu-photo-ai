import cv2
import numpy as np
import mediapipe as mp

# Mediapipeのランドマークインデックス
# これらを定義しておくとコードが読みやすくなる
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 130
LEFT_EYE_RIGHT = 133 # 33から変更して内側の点を取得
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362 # 263から変更
RIGHT_EYE_RIGHT = 359


class FaceAnalyzer:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def _calculate_ear(self, landmarks, eye_top_idx, eye_bottom_idx, eye_left_idx, eye_right_idx):
        """Eye Aspect Ratio (EAR)を計算する"""
        eye_top = landmarks[eye_top_idx]
        eye_bottom = landmarks[eye_bottom_idx]
        eye_left = landmarks[eye_left_idx]
        eye_right = landmarks[eye_right_idx]

        ver_dist = np.linalg.norm([eye_top.x - eye_bottom.x, eye_top.y - eye_bottom.y])
        hor_dist = np.linalg.norm([eye_left.x - eye_right.x, eye_left.y - eye_right.y])

        return ver_dist / hor_dist if hor_dist > 0 else 0

    def analyze(self, image):
        """画像を受け取り、全ての分析結果を返す"""
        img_bgr = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = img_rgb.shape

        results = self.face_mesh.process(img_rgb)
        annotated_image = img_rgb.copy()

        if not results.multi_face_landmarks:
            return None # 顔が検出できなかった場合はNoneを返す

        face_landmarks = results.multi_face_landmarks[0].landmark

        # --- 1. 明るさ (顔領域の平均輝度) ---
        x_coords = [lm.x for lm in face_landmarks]
        y_coords = [lm.y for lm in face_landmarks]
        x_min, x_max = int(min(x_coords) * width), int(max(x_coords) * width)
        y_min, y_max = int(min(y_coords) * height), int(max(y_coords) * height)
        face_roi = cv2.cvtColor(img_rgb[y_min:y_max, x_min:x_max], cv2.COLOR_RGB2GRAY)
        brightness = np.mean(face_roi) if face_roi.size > 0 else 0

        # --- 2. 笑顔スコア (口角 + 目の細まり具合) ---
        # 口のスコア
        mouth_width = np.linalg.norm([face_landmarks[MOUTH_LEFT].x - face_landmarks[MOUTH_RIGHT].x, face_landmarks[MOUTH_LEFT].y - face_landmarks[MOUTH_RIGHT].y])
        eye_dist = np.linalg.norm([face_landmarks[LEFT_EYE_LEFT].x - face_landmarks[RIGHT_EYE_RIGHT].x, face_landmarks[LEFT_EYE_LEFT].y - face_landmarks[RIGHT_EYE_RIGHT].y])
        mouth_score = mouth_width / eye_dist if eye_dist > 0 else 0
        
        # 目のスコア (EAR: 小さいほど目が細い = 笑っている)
        left_ear = self._calculate_ear(face_landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
        right_ear = self._calculate_ear(face_landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
        # EARは小さい方が良いため、逆数をとるなどしてスコア化する。ここでは単純化のため、正規化されたmouth_scoreと組み合わせる。
        # 0.25は平常時のEARの目安。それより小さいほど高スコアになるように調整
        eye_smile_score = max(0, 1 - ( (left_ear + right_ear) / 2 / 0.25) ) 

        # 総合笑顔スコア（口の比重を大きくする）
        smile_score = (mouth_score * 0.7) + (eye_smile_score * 0.3)
        # スコアの内訳も返す
        smile_metrics = {"mouth": mouth_score, "eyes": eye_smile_score}

        # --- 3. 顔の傾き ---
        left_eye_y = face_landmarks[LEFT_EYE_TOP].y
        right_eye_y = face_landmarks[RIGHT_EYE_TOP].y
        tilt_score = abs(left_eye_y - right_eye_y)

        # --- 描画処理 ---
        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        analysis_data = {
            "brightness": brightness,
            "smile_score": smile_score,
            "smile_metrics": smile_metrics,
            "tilt_score": tilt_score,
            "annotated_image": annotated_image
        }
        return analysis_data