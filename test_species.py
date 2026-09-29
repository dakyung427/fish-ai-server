import cv2
import json

from vision.detector import FishDetector
from services.species_service import SpeciesService


VIDEO_PATH = "goldfish.mp4"


# ==========================================
# 테스트용 영상 클래스
# SpeciesService는 camera.read() 형태를 사용하므로
# MP4도 같은 방식으로 읽을 수 있게 만듦
# ==========================================
class VideoCamera:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError("영상 열기 실패")

    def read(self):
        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def stop(self):
        self.cap.release()


# ==========================================
# 영상 준비
# ==========================================
camera = VideoCamera(VIDEO_PATH)


# ==========================================
# YOLO + ByteTrack 준비
# ==========================================
detector = FishDetector()


# ==========================================
# 어종 분석 서비스
# 3초 동안 분석
# ==========================================
species_service = SpeciesService(
    detector=detector,
    analysis_time=3.0
)


# ==========================================
# 어종 분석 실행
# ==========================================
result = species_service.analyze(camera)


# ==========================================
# 최종 결과 JSON 출력
# ==========================================
print("\n========== 최종 어종 분석 결과 ==========")

print(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2
    )
)


# ==========================================
# 종료
# ==========================================
camera.stop()