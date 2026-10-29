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


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.get('/activity', response_class=HTMLResponse)
async def activity_page():
    return '''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AI Activity Monitor</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; padding: 24px; background: #101827; color: #f3f5f8;
           font-family: Arial, sans-serif; }
    main { max-width: 1100px; margin: auto; }
    h1 { font-size: 26px; margin: 0 0 8px; }
    p { color: #a7b4c7; margin: 6px 0 18px; }
    .video { background: #050a12; border: 1px solid #344256;
             border-radius: 10px; overflow: hidden; }
    img { width: 100%; display: block; min-height: 200px; object-fit: contain; }
    .hint { font-size: 13px; margin-top: 16px; }
  </style>
</head>
<body><main>
  <h1>AI Activity Monitor</h1>
  <p>YOLO11 + CSRT + HSV + SOM · Real-time fish tracking</p>
  <div class="video"><img src="/activity/stream" alt="실시간 추적 분석 영상"></div>
  <p class="hint">Green: NORMAL · Red: WARNING · SOM window: 10 seconds</p>
</main></body></html>'''


@app.get('/activity/stream')
async def activity_stream():
    async def generate():
        last_frame = None
        while True:
            frame = activity_service.preview_snapshot() if activity_service else None
            if frame is not None and frame is not last_frame:
                last_frame = frame
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n'
                       b'Cache-Control: no-store\r\n\r\n' + frame + b'\r\n')
            await asyncio.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-store'},
    )


@app.on_event('startup')
async def startup_event():
    global camera, detector, species_service, activity_tracker
    global activity_service, activity_task, ws_client, ws_task, yolo_gate

    print('AI 서버 시작')
    camera = RTSPCamera()
    camera.start()
    yolo_gate = YOLOPriorityGate()

    detector = FishDetector()
    original_detect = detector.detect

    def safe_detect(frame):
        with yolo_gate.low():
            return original_detect(frame)

    detector.detect = safe_detect
    species_service = SpeciesService(detector=detector, analysis_time=3.0)
    print('[SPECIES] ready')

    activity_tracker = FishCSRTTracker(
        weights_path='weights/best.pt', yolo_gate=yolo_gate
    )
    activity_service = ActivityService(
        tracker=activity_tracker, camera=camera,
        model_path='weights/som_model.npz'
    )
    print('[ACTIVITY] ready')

    ws_client = WebSocketClient(
        species_service=species_service,
        activity_service=activity_service,
        camera=camera,
    )
    ws_task = asyncio.create_task(ws_client.run())
    activity_task = asyncio.create_task(asyncio.to_thread(activity_service.run_sync))
    print('AI 서버 초기화 완료')


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
            await asyncio.wait_for(activity_task, timeout=5.0)
        except asyncio.TimeoutError:
            print('[ACTIVITY] 분석 스레드 종료 대기 초과')
        except asyncio.CancelledError:
            pass
    if camera is not None:
        camera.stop()
    print('AI 서버 종료 완료')
