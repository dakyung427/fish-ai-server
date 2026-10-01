import os

from dotenv import load_dotenv


# ==========================================
# .env 파일 불러오기
# ==========================================
load_dotenv()


# ==========================================
# 영상 입력 설정
# ==========================================

# 실제 라즈베리파이 RTSP 주소
RTSP_URL = os.getenv(
    "RTSP_URL",
    "rtsp://192.168.0.25:8554/cam"
)

# 테스트 영상 사용 여부
# true  → MP4 사용
# false → RTSP 사용
USE_TEST_VIDEO = os.getenv(
    "USE_TEST_VIDEO",
    "false"
).lower() == "true"

# 테스트용 MP4 경로
TEST_VIDEO_PATH = os.getenv(
    "TEST_VIDEO_PATH",
    "goldfish.mp4"
)


# ==========================================
# WebSocket 설정
# ==========================================

# Node.js WebSocket 서버 주소
NODE_WS_URL = os.getenv(
    "NODE_WS_URL",
    "ws://localhost:3000"
)


# ==========================================
# YOLO / ByteTrack 설정
# ==========================================

# YOLO 모델 경로
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    "weights/best.pt"
)

# ByteTrack 설정 파일
BYTETRACK_CONFIG = os.getenv(
    "BYTETRACK_CONFIG",
    "trackers/custom_bytetrack.yaml"
)


# ==========================================
# 어종 한글 이름
# ==========================================

SPECIES_NAME_KO = {
    "AngelFish": "엔젤피쉬",
    "BlueTang": "블루탱",
    "ButterflyFish": "나비고기",
    "ClownFish": "흰동가리",
    "GoldFish": "금붕어",
    "Gourami": "구라미",
    "MorishIdol": "무어리쉬 아이돌",
    "PlatyFish": "플래티",
    "RibbonedSweetlips": "리본드 스위트립스",
    "ThreeStripedDamselfish": "세줄자리돔",
    "YellowCichlid": "옐로 시클리드",
    "YellowTang": "옐로탱",
    "ZebraFish": "제브라피쉬",
}