from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
from datetime import datetime
import cv2
import threading

app = FastAPI()

# 모델
species_model = YOLO("yolo11s.pt")      # 어종 분석
activity_model = YOLO("yolo11s2.pt")    # 활동성 분석

# 상태
is_running = False
worker_thread = None
current_rtsp_url = None

# 결과 저장
species_result = {
    "totalFishCount": 0,
    "detections": [],
    "updatedAt": None
}

activity_result = {
    "activityStatus": "분석 전",
    "anomalyScore": 0,
    "updatedAt": None
}


class StartRequest(BaseModel):
    rtspUrl: str


# 활동성 분석
def rtsp_loop(rtsp_url):
    global is_running, activity_result

    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        is_running = False
        return

    frame_count = 0

    while is_running:
        ret, frame = cap.read()

        if not ret:
            continue

        frame_count += 1

        # 30프레임마다 활동성 분석
        if frame_count % 30 == 0:

            results = activity_model(
                frame,
                conf=0.5,
                verbose=False
            )

            # 추후 SOM 연결 위치
            # features 생성
            # ↓
            # SOM 예측
            # ↓
            # anomalyScore 계산

            activity_result = {
                "activityStatus": "정상",
                "anomalyScore": 0.1,
                "updatedAt": datetime.now().isoformat(timespec="seconds")
            }

    cap.release()


# AI 분석 시작
@app.post("/api/ai/analysis/start")
def start_analysis(request: StartRequest):
    global is_running, worker_thread, current_rtsp_url

    if is_running:
        return {
            "result": "fail",
            "message": "이미 실행 중"
        }

    current_rtsp_url = request.rtspUrl
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


# 어종 분석
@app.post("/api/ai/species")
def species_analysis():
    global species_result

    cap = cv2.VideoCapture(current_rtsp_url)

    ret, frame = cap.read()

    cap.release()

    if not ret:
        return {
            "result": "fail",
            "message": "프레임 읽기 실패"
        }

    results = species_model(
        frame,
        conf=0.5,
        verbose=False
    )

    species_count = {}
    confidence_sum = {}

    total_count = 0

    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = species_model.names[cls]

        if class_name not in species_count:
            species_count[class_name] = 0
            confidence_sum[class_name] = 0

        species_count[class_name] += 1
        confidence_sum[class_name] += conf

        total_count += 1


    detections = []

    for class_name, count in species_count.items():

        avg_confidence = (
            confidence_sum[class_name] / count
        )

        detections.append({
            "className": class_name,
            "count": count,
            "confidence": round(avg_confidence, 2)
        })


    species_result = {
        "totalFishCount": total_count,
        "detections": detections,
        "updatedAt": datetime.now().isoformat(timespec="seconds")
    }


    return {
        "result": "success",
        "analysis": species_result
    }


# 어종 결과 조회
@app.get("/api/ai/species/result")
def get_species_result():

    return {
        "result": "success",
        "analysis": species_result
    }


# 활동성 결과 조회
@app.get("/api/ai/activity/result")
def get_activity_result():

    return {
        "result": "success",
        "analysis": activity_result
    }


# 분석 종료
@app.post("/api/ai/analysis/stop")
def stop_analysis():

    global is_running

    is_running = False

    return {
        "result": "success",
        "message": "분석 종료"
    }