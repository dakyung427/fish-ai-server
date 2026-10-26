from ultralytics import YOLO

from config import YOLO_MODEL_PATH, BYTETRACK_CONFIG


class FishDetector:
    def __init__(self, conf=0.2):
        print("YOLO 모델 로딩...")

        self.model = YOLO(YOLO_MODEL_PATH)
        self.conf = conf

        print("YOLO 모델 로딩 완료")

    # ==========================================
    # YOLO + ByteTrack 실행
    # ==========================================
    def detect(self, frame):

        # RTSP 원본 영상 해상도 가져오기
        height, width = frame.shape[:2]

        # 원본 해상도를 기준으로 YOLO 분석
        results = self.model.track(
            frame,
            persist=True,
            tracker=BYTETRACK_CONFIG,
            conf=self.conf,
            imgsz=(height, width),
            verbose=False
        )

        return results