from ultralytics import YOLO

from config import YOLO_MODEL_PATH, BYTETRACK_CONFIG


class FishDetector:
    def __init__(self, conf=0.2):
        print("YOLO 모델 로딩...")

        self.model = YOLO(YOLO_MODEL_PATH)
        self.conf = conf

        print("YOLO 모델 로딩 완료")

    # YOLO + ByteTrack 실행
    def detect(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker=BYTETRACK_CONFIG,
            conf=self.conf,
            verbose=False
        )

        return results