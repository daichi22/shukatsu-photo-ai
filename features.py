import cv2
import numpy as np
import mediapipe as mp
import yaml
from dataclasses import dataclass

# --- 初期設定 ---
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# --- データクラス定義 ---
@dataclass
class AnalysisResult:
    label: str; value: float; status: str; message_key: str; normalized_score: int

# --- 定数定義 ---
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER = 159, 145, 130, 33
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER = 386, 374, 362, 263

# --- メイン分析クラス ---
class FaceAnalyzer:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )

    def _normalize(self, value, ideal, worst, is_range=False, range_min=0, range_max=0, higher_is_better=True):
        if is_range:
            center = (range_min + range_max) / 2
            dist = abs(value - center)
            max_dist = (range_max - range_min) / 2
            score = (max_dist - dist) / max_dist * 100 if max_dist > 0 else 100
        else:
            if higher_is_better:
                if ideal == worst: return 100 if value >= ideal else 0
                score = ((value - worst) / (ideal - worst)) * 100
            else:
                if ideal == worst: return 100 if value <= ideal else 0
                score = ((worst - value) / (worst - ideal)) * 100
        return max(0, min(100, int(score)))

    # 各評価項目を判定・正規化するプライベートメソッド群
    def _evaluate_brightness(self, value):
        cfg = config['brightness']
        status = "OK" if cfg['ideal_min'] <= value <= cfg['ideal_max'] else "ERROR"
        score = self._normalize(value, 0, 0, is_range=True, range_min=cfg['ideal_min'], range_max=cfg['ideal_max'])
        return AnalysisResult("顔の明るさ", value, status, "BRIGHTNESS", score)

    def _evaluate_smile(self, value):
        cfg = config['smile']
        status = "OK" if value >= cfg['ideal_min'] else "WARN"
        score = self._normalize(value, cfg['ideal_value'], cfg['worst_value'])
        return AnalysisResult("笑顔スコア", value, status, "SMILE", score)
    
    def _evaluate_tilt(self, value):
        cfg = config['tilt']
        status = "OK" if value <= cfg['ideal_max'] else "WARN"
        score = self._normalize(value, cfg['ideal_value'], cfg['worst_value'], higher_is_better=False)
        return AnalysisResult("顔の傾き", value, status, "TILT", score)

    def _evaluate_face_ratio(self, value):
        cfg = config['composition']
        status = "OK" if cfg['face_ratio_min'] <= value <= cfg['face_ratio_max'] else "WARN"
        score = self._normalize(abs(value - cfg['face_ratio_ideal']), 0, abs(cfg['face_ratio_worst'] - cfg['face_ratio_ideal']), higher_is_better=False)
        return AnalysisResult("顔の比率", value, status, "FACE_RATIO", score)
    
    def _evaluate_center_offset(self, value):
        cfg = config['composition']
        status = "OK" if value <= cfg['center_offset_max'] else "WARN"
        score = self._normalize(value, cfg['center_offset_ideal'], cfg['center_offset_worst'], higher_is_better=False)
        return AnalysisResult("中心位置", value, status, "CENTER_OFFSET", score)

    def _evaluate_sharpness(self, value):
        cfg = config['sharpness']
        status = "OK" if value >= cfg['laplacian_variance_min'] else "ERROR"
        score = self._normalize(value, cfg['laplacian_variance_ideal'], cfg['laplacian_variance_worst'])
        return AnalysisResult("写真の鮮明度", value, status, "SHARPNESS", score)

    def analyze(self, image):
        img_pil = image.convert('RGB')
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = img_rgb.shape
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks: return None

        landmarks = results.multi_face_landmarks[0].landmark
        
        unique_indices = set(i for conn in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in conn)
        face_points = np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in unique_indices], dtype=np.int32)
        hull = cv2.convexHull(face_points)
        face_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, hull, 255)

        # 生データの計算
        raw_brightness = cv2.mean(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), mask=face_mask)[0]
        raw_smile_score = self._calculate_smile_score(landmarks)
        raw_tilt_score = abs(landmarks[LEFT_EYE_TOP].y - landmarks[RIGHT_EYE_TOP].y)
        face_rect = cv2.boundingRect(hull)
        raw_face_ratio = face_rect[3] / height
        face_center = (face_rect[0] + face_rect[2] / 2, face_rect[1] + face_rect[3] / 2)
        raw_center_offset = np.linalg.norm([face_center[0] - width/2, face_center[1] - height/2]) / np.linalg.norm([width, height])
        raw_sharpness = self._calculate_sharpness(img_bgr, face_mask)

        # 構造化された結果を生成
        analysis_results = {
            "brightness": self._evaluate_brightness(raw_brightness),
            "smile": self._evaluate_smile(raw_smile_score),
            "tilt": self._evaluate_tilt(raw_tilt_score),
            "face_ratio": self._evaluate_face_ratio(raw_face_ratio),
            "center_offset": self._evaluate_center_offset(raw_center_offset),
            "sharpness": self._evaluate_sharpness(raw_sharpness)
        }
        
        # 総合スコアの計算
        total_score = 0
        total_weight = 0
        weights = config['scoring_weights']
        for key, result in analysis_results.items():
            weight = weights.get(key, 0)
            total_score += result.normalized_score * weight
            total_weight += weight
        final_score = total_score / total_weight if total_weight > 0 else 0

        annotated_image = self._draw_annotations(img_rgb, hull, landmarks)
        
        return {
            "results": analysis_results,
            "annotated_image": annotated_image,
            "final_score": final_score
        }

    def _calculate_ear(self, landmarks, eye_top_idx, eye_bottom_idx, eye_left_idx, eye_right_idx):
        eye_top = landmarks[eye_top_idx]; eye_bottom = landmarks[eye_bottom_idx]
        eye_left = landmarks[eye_left_idx]; eye_right = landmarks[eye_right_idx]
        ver_dist = np.linalg.norm([eye_top.x - eye_bottom.x, eye_top.y - eye_bottom.y])
        hor_dist = np.linalg.norm([eye_left.x - eye_right.x, eye_left.y - eye_right.y])
        return ver_dist / hor_dist if hor_dist > 0 else 0

    def _calculate_smile_score(self, landmarks):
        mouth_width = np.linalg.norm([landmarks[MOUTH_LEFT].x - landmarks[MOUTH_RIGHT].x, landmarks[MOUTH_LEFT].y - landmarks[MOUTH_RIGHT].y])
        eye_dist = np.linalg.norm([landmarks[LEFT_EYE_LEFT_CORNER].x - landmarks[RIGHT_EYE_RIGHT_CORNER].x, landmarks[LEFT_EYE_LEFT_CORNER].y - landmarks[RIGHT_EYE_RIGHT_CORNER].y])
        mouth_score = mouth_width / eye_dist if eye_dist > 0 else 0
        left_ear = self._calculate_ear(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER)
        right_ear = self._calculate_ear(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER)
        avg_ear = (left_ear + right_ear) / 2
        eye_smile_score = max(0, 1 - (avg_ear / config['smile']['eye_aspect_ratio_base']))
        return (mouth_score * config['smile']['mouth_weight']) + (eye_smile_score * config['smile']['eyes_weight'])

    def _calculate_sharpness(self, img_bgr, face_mask):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_face = laplacian[face_mask == 255]
        return laplacian_face.var() if laplacian_face.size > 0 else 0

    def _draw_annotations(self, img_rgb, hull, landmarks):
        annotated_image = img_rgb.copy()
        cv2.line(annotated_image, (0, int(landmarks[LEFT_EYE_TOP].y * img_rgb.shape[0])), (img_rgb.shape[1], int(landmarks[LEFT_EYE_TOP].y * img_rgb.shape[0])), (255, 0, 0), 1)
        cv2.line(annotated_image, (0, int(landmarks[RIGHT_EYE_TOP].y * img_rgb.shape[0])), (img_rgb.shape[1], int(landmarks[RIGHT_EYE_TOP].y * img_rgb.shape[0])), (255, 0, 0), 1)
        cv2.polylines(annotated_image, [hull], isClosed=True, color=(0, 255, 0), thickness=2)
        return annotated_image