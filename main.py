
import asyncio

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from core.yolo_priority import YOLOPriorityGate
from vision.rtsp_camera import RTSPCamera
from vision.detector import FishDetector
from vision.detector_csrt import FishCSRTTracker
from services.species_service import SpeciesService
from services.activity_service import ActivityService
from websocket.ws_client import WebSocketClient


app = FastAPI(title='Fish AI Server')


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
@app.get('/health')
async def health():
    return {'status': 'ok'}


# ==========================================
# 활동성 실시간 영상 페이지
# 영상만 표시
# ==========================================
@app.get('/activity', response_class=HTMLResponse)
async def activity_page():
    return '''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AI Activity Monitor</title>

  <style>
    * {
      box-sizing: border-box;
    }

    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #101827;
    }

    img {
      display: block;
      width: 100%;
      height: 100vh;
      object-fit: contain;
    }
  </style>
</head>

<body>
  <img
    src="/activity/stream"
    alt="실시간 추적 영상"
  >
</body>
</html>'''


# ==========================================
# 활동성 실시간 MJPEG 스트리밍
# ==========================================
@app.get('/activity/stream')
async def activity_stream():

    async def generate():

        last_frame = None

        while True:

            frame = (
                activity_service.preview_snapshot()
                if activity_service
                else None
            )

            if frame is not None and frame is not last_frame:

                last_frame = frame

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Cache-Control: no-store\r\n\r\n'
                    + frame
                    + b'\r\n'
                )

            await asyncio.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-store'},
    )


# ==========================================
# FastAPI 시작 시 실행
# ==========================================
@app.on_event('startup')
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

    print('AI 서버 시작')

    # ======================================
    # RTSP 카메라 시작
    # ======================================
    camera = RTSPCamera()
    camera.start()

    # ======================================
    # YOLO 우선순위 제어
    # ======================================
    yolo_gate = YOLOPriorityGate()

    # ======================================
    # 어종 분석 YOLO + ByteTrack
    # ======================================
    detector = FishDetector()

    original_detect = detector.detect

    def safe_detect(frame):

        with yolo_gate.low():
            return original_detect(frame)

    detector.detect = safe_detect

    species_service = SpeciesService(
        detector=detector,
        analysis_time=3.0
    )

    print('[SPECIES] ready')

    # ======================================
    # 활동성 YOLO + CSRT 추적
    # ======================================
    activity_tracker = FishCSRTTracker(
        weights_path='weights/best.pt',
        yolo_gate=yolo_gate
    )

    # ======================================
    # SOM 활동성 이상탐지
    # ======================================
    activity_service = ActivityService(
        tracker=activity_tracker,
        camera=camera,
        model_path='weights/som_model.npz'
    )

    print('[ACTIVITY] ready')

    # ======================================
    # WebSocket 서버 연동
    # ======================================
    ws_client = WebSocketClient(
        species_service=species_service,
        activity_service=activity_service,
        camera=camera,
    )

    ws_task = asyncio.create_task(
        ws_client.run()
    )

    # ======================================
    # 활동성 분석 별도 스레드
    # ======================================
    activity_task = asyncio.create_task(
        asyncio.to_thread(
            activity_service.run_sync
        )
    )

    print('AI 서버 초기화 완료')


# ==========================================
# FastAPI 종료 시 실행
# ==========================================
@app.on_event('shutdown')
async def shutdown_event():

    print('AI 서버 종료 중...')

    if activity_service is not None:
        activity_service.stop()

    if ws_client is not None:
        await ws_client.stop()

    if ws_task is not None:

        ws_task.cancel()

        try:
            await ws_task
        except asyncio.CancelledError:
            pass

    if activity_task is not None:

        try:
            await asyncio.wait_for(
                activity_task,
                timeout=5.0
            )

        except asyncio.TimeoutError:
            print(
                '[ACTIVITY] 분석 스레드 종료 대기 초과'
            )

        except asyncio.CancelledError:
            pass

    if camera is not None:
        camera.stop()

    print('AI 서버 종료 완료')
