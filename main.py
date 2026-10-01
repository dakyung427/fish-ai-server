import asyncio

from fastapi import FastAPI

from config import (
    USE_TEST_VIDEO,
    TEST_VIDEO_PATH
)

from vision.rtsp_camera import RTSPCamera
from vision.video_camera import VideoCamera
from vision.detector import FishDetector

from services.species_service import SpeciesService

from websocket.ws_client import WebSocketClient


app = FastAPI(
    title="Fish AI Server"
)


camera = None
detector = None
species_service = None
ws_client = None
ws_task = None


# ==========================================
# 서버 상태 확인
# ==========================================
@app.get("/health")
async def health():
    return {
        "status": "ok"
    }


# ==========================================
# FastAPI 시작 시 실행
# ==========================================
@app.on_event("startup")
async def startup_event():

    global camera
    global detector
    global species_service
    global ws_client
    global ws_task

    print("AI 서버 시작")


    # ======================================
    # 영상 입력 선택
    # ======================================
    if USE_TEST_VIDEO:

        print("MP4 테스트 영상 모드")

        camera = VideoCamera(
            TEST_VIDEO_PATH
        )

        camera.start()

    else:

        print("RTSP 실시간 영상 모드")

        camera = RTSPCamera()

        camera.start()


    # ======================================
    # YOLO + ByteTrack
    # ======================================
    detector = FishDetector()


    # ======================================
    # 어종 분석 서비스
    # ======================================
    species_service = SpeciesService(
        detector=detector,
        analysis_time=3.0
    )


    # ======================================
    # Node.js WebSocket 연결
    # ======================================
    ws_client = WebSocketClient(
        species_service=species_service,
        camera=camera
    )


    # WebSocket 클라이언트를
    # FastAPI와 동시에 백그라운드 실행
    ws_task = asyncio.create_task(
        ws_client.run()
    )

    print("AI 서버 초기화 완료")


# ==========================================
# FastAPI 종료 시 실행
# ==========================================
@app.on_event("shutdown")
async def shutdown_event():

    global camera
    global ws_client
    global ws_task

    print("AI 서버 종료 중...")


    if ws_client is not None:
        await ws_client.stop()


    if ws_task is not None:
        ws_task.cancel()


    if camera is not None:
        camera.stop()


    print("AI 서버 종료 완료")