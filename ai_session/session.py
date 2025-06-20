# ai_session/session.py
import time
import base64
import numpy as np
import cv2
from collision_predictor import CollisionPredictor
from collision_predictor.yolo_utils import detect_bboxes

class AISession:
    def __init__(self, user_id: str):
        self.user_id = user_id
        # 모델 로드
        self.predictor = CollisionPredictor(
            model_path="assets/first_best_model.pth",
            threshold=0.55 # 임계값 -> 이진 분류 임계값값
        )

        # 메타데이터 초기화
        self.start_time = time.time()
        self.frame_count = 0

    def process_image(self, img: np.ndarray) -> dict:
        """
        이미지를 받아서 예측을 수행하고,
        내부 카운터와 타임스탬프 기반 반환값을 계산합니다.
        """

        
        self.frame_count += 1
        timestamp = time.time() - self.start_time

        # 예: YOLO 바운딩박스 검출
        bboxes = detect_bboxes(img, weights_path="assets/last11.pt")
        prob = self.predictor.update(
            bboxes_xyxy=bboxes,
            img_w=1280,
            img_h=720,
        )

        return self.predictor.is_danger(prob)

    def get_usage(self) -> dict:
        """세션 종료 시 호출 → 사용 메타데이터 반환"""
        end_time = time.time()
        return {
            "user_id": self.user_id,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
            "photos": self.frame_count,  # ✅ API 요구사항에 맞게 이름 수정
        }
