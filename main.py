
import asyncio

from fastapi import FastAPI

from core.yolo_priority import YOLOPriorityGate

from vision.rtsp_camera import RTSPCamera
from vision.detector import FishDetector
from vision.detector_csrt import FishCSRTTracker

from services.species_service import SpeciesService
from services.activity_service import ActivityService

from websocket.ws_client import WebSocketClient


app = FastAPI(
    title="Fish AI Server"
)


camera = None

detector = None
species_service = None

activity_tracker = None
activity_service = None
activity_task = None

ws_client = None
ws_task = None

yolo_gate = None


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
    global activity_tracker
    global activity_service
    global activity_task
    global ws_client
    global ws_task
    global yolo_gate

    print("AI 서버 시작")

    # ======================================
    # RTSP 실시간 영상 연결
    # ======================================
    print("RTSP 실시간 영상 모드")

    camera = RTSPCamera()
    camera.start()

    # ======================================
    # YOLO 우선순위 제어
    # ======================================
    yolo_gate = YOLOPriorityGate()

    # ======================================
    # 기존 YOLO + ByteTrack
    # ======================================
    detector = FishDetector()

    # 갱신하기 YOLO는 낮은 우선순위
    original_detect = detector.detect

    def safe_detect(frame):
        with yolo_gate.low():
            return original_detect(frame)

    detector.detect = safe_detect

    # ======================================
    # 기존 어종·개체수 분석 서비스
    # ======================================
    species_service = SpeciesService(
        detector=detector,
        analysis_time=3.0
    )

    print("[SPECIES] 어종 분석 준비 완료")

    # ======================================
    # YOLO + CSRT + HSV 추적기
    # ======================================
    activity_tracker = FishCSRTTracker(
        weights_path="weights/best.pt",
        yolo_gate=yolo_gate
    )

    print("[TRACKER] CSRT 추적기 준비 완료")

    # ======================================
    # SOM 활동성 이상탐지 서비스
    # ======================================
    activity_service = ActivityService(
        tracker=activity_tracker,
        camera=camera,
        model_path="weights/som_model.npz"
    )

    print("[ACTIVITY] SOM 분석 준비 완료")

    # ======================================
    # Node.js WebSocket 연결
    # ======================================
    ws_client = WebSocketClient(
        species_service=species_service,
        activity_service=activity_service,
        camera=camera
    )

    # WebSocket 클라이언트 실행
    ws_task = asyncio.create_task(
        ws_client.run()
    )

    # ======================================
    # 활동성 분석 별도 스레드 실행
    # ======================================
    activity_task = asyncio.create_task(
        asyncio.to_thread(
            activity_service.run_sync
        )
    )

    print("AI 서버 초기화 완료")


# ==========================================
# FastAPI 종료 시 실행
# ==========================================
@app.on_event("shutdown")
async def shutdown_event():

    global camera
    global activity_service
    global activity_task
    global ws_client
    global ws_task

    print("AI 서버 종료 중...")

    # ======================================
    # 활동성 분석 종료 요청
    # ======================================
    if activity_service is not None:
        activity_service.stop()

    # ======================================
    # WebSocket 종료
    # ======================================
    if ws_client is not None:
        await ws_client.stop()

    if ws_task is not None:
        ws_task.cancel()

        try:
            await ws_task
        except asyncio.CancelledError:
            pass

    # ======================================
    # RTSP 카메라 종료
    # ======================================
    if camera is not None:
        camera.stop()

    # ======================================
    # 활동성 분석 종료 대기
    # ======================================
    if activity_task is not None:
        try:
            await asyncio.wait_for(
                activity_task,
                timeout=5.0
            )
        except asyncio.TimeoutError:
            print("[ACTIVITY] 종료 대기 시간 초과")
        except asyncio.CancelledError:
            pass

    print("AI 서버 종료 완료")
