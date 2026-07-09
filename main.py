from fastapi import FastAPI
from pydantic import BaseModel
import cv2
import threading
import time

app = FastAPI()

is_running = False
worker_thread = None


class StartRequest(BaseModel):
    rtspUrl: str


def rtsp_loop(rtsp_url: str):
    global is_running

    print("[RTSP] 연결 시도:", rtsp_url)

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("[RTSP] 연결 실패")
        is_running = False
        return

    print("[RTSP] 연결 성공")

    frame_count = 0

    while is_running:
        ret, frame = cap.read()

        if not ret:
            print("[RTSP] 프레임 읽기 실패")
            time.sleep(1)
            continue

        frame_count += 1

        if frame_count % 30 == 0:
            print("[RTSP] 프레임 수신 중:", frame_count)

    cap.release()
    print("[RTSP] 분석 종료")


@app.get("/api/ai/health")
def health():
    return {
        "result": "success",
        "status": "running"
    }


@app.post("/api/ai/analysis/start")
def start_analysis(request: StartRequest):
    global is_running, worker_thread

    if is_running:
        return {
            "result": "fail",
            "message": "이미 분석 중입니다."
        }

    is_running = True

    worker_thread = threading.Thread(
        target=rtsp_loop,
        args=(request.rtspUrl,),
        daemon=True
    )
    worker_thread.start()

    return {
        "result": "success",
        "message": "AI 분석 시작"
    }


@app.post("/api/ai/analysis/stop")
def stop_analysis():
    global is_running

    is_running = False

    return {
        "result": "success",
        "message": "AI 분석 중지"
    }


@app.get("/api/ai/analysis/status")
def analysis_status():
    return {
        "isRunning": is_running
    }