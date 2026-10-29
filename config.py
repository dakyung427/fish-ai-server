import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 영상 입력 설정 (RTSP / MP4)
# ==========================================

USE_TEST_VIDEO = os.getenv("USE_TEST_VIDEO", "false").lower() == "true"
TEST_VIDEO_PATH = os.getenv("TEST_VIDEO_PATH", "goldfish.mp4")

# ==========================================
# RTSP / WebSocket 설정
# ==========================================

RTSP_URL = os.getenv("RTSP_URL")
NODE_WS_URL = os.getenv("NODE_WS_URL")

# ==========================================
# YOLO / ByteTrack 설정
# ==========================================

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
BYTETRACK_CONFIG = os.getenv("BYTETRACK_CONFIG")

# ==========================================
# 어종 한글 이름
# ==========================================

SPECIES_NAME_KO = {
    "AngelFish": "엔젤피쉬",
    "BlueTang": "블루탱",
    "ButterflyFish": "나비고기",
    "ClownFish": "흰동가리",
    "GoldFish": "금붕어",
    "FishModel": "돌고래",  
    "Gourami": "구라미",
    "MorishIdol": "무어리쉬 아이돌",
    "PlatyFish": "플래티",
    "RibbonedSweetlips": "리본드 스위트립스",
    "ThreeStripedDamselfish": "세줄자리돔",
    "YellowCichlid": "옐로 시클리드",
    "YellowTang": "옐로탱",
    "ZebraFish": "제브라피쉬",
}