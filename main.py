from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
from datetime import datetime

import cv2
import threading
import time

app = FastAPI()

# YOLO 모델 불러오기
model = YOLO("yolo11s.pt")

# 분석 상태
is_running = False
worker_thread = None
current_rtsp_url = None
status_updated_at = None

# 로그 비교용: confidence 제외하고 비교
last_logged_detection_state = None

# 최신 분석 결과
latest_analysis = {
    "fishCount": 0,
    "fishStatus": "분석 전",
    "detections": [],
    "updatedAt": None
}


class StartRequest(BaseModel):
    rtspUrl: str


def rtsp_loop(rtsp_url: str):
    global is_running
    global latest_analysis
    global last_logged_detection_state

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

        # 30프레임마다 YOLO 분석
        if frame_count % 30 == 0:
            results = model(frame, conf=0.5, verbose=False)
            boxes = results[0].boxes

            class_summary = {}
            total_count = 0

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                if class_name not in class_summary:
                    class_summary[class_name] = {
                        "count": 0,
                        "confidence_sum": 0.0
                    }

                class_summary[class_name]["count"] += 1
                class_summary[class_name]["confidence_sum"] += confidence
                total_count += 1

            detections = []

            for class_name, data in class_summary.items():
                avg_confidence = data["confidence_sum"] / data["count"]

                detections.append({
                    "className": class_name,
                    "count": data["count"],
                    "confidence": round(avg_confidence, 2)
                })

            detections.sort(key=lambda x: x["className"])

            latest_analysis = {
                "fishCount": total_count,
                "fishStatus": "정상" if total_count > 0 else "미탐지",
                "detections": detections,
                "updatedAt": datetime.now().isoformat(timespec="seconds")
            }

            # confidence는 제외하고 개수/종류가 바뀔 때만 로그 출력
            detection_state_for_log = {
                "fishCount": total_count,
                "fishStatus": latest_analysis["fishStatus"],
                "detections": [
                    {
                        "className": item["className"],
                        "count": item["count"]
                    }
                    for item in detections
                ]
            }

            if detection_state_for_log != last_logged_detection_state:
                print("[YOLO] 결과 변경:", latest_analysis)
                last_logged_detection_state = detection_state_for_log

            # 시연용 OpenCV 박스 영상 창
            annotated_frame = results[0].plot()
            cv2.imshow("AI Detection", annotated_frame)

            # q 누르면 분석 중지
            if cv2.waitKey(1) & 0xFF == ord("q"):
                is_running = False
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[RTSP] 분석 종료")


@app.get("/api/ai/health")
def health():
    return {
        "result": "success",
        "status": "running"
    }


@app.post("/api/ai/analysis/start")
def start_analysis(request: StartRequest):
    global is_running
    global worker_thread
    global current_rtsp_url
    global status_updated_at
    global last_logged_detection_state

    if is_running:
        return {
            "result": "fail",
            "message": "이미 분석 중입니다."
        }

    is_running = True
    current_rtsp_url = request.rtspUrl
    status_updated_at = datetime.now().isoformat(timespec="seconds")
    last_logged_detection_state = None

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
    global status_updated_at

    is_running = False
    status_updated_at = datetime.now().isoformat(timespec="seconds")

    return {
        "result": "success",
        "message": "AI 분석 중지"
    }


@app.get("/api/ai/analysis/status")
def analysis_status():
    return {
        "result": "success",
        "analysis": {
            "isRunning": is_running,
            "rtspUrl": current_rtsp_url,
            "updatedAt": status_updated_at
        }
    }


@app.get("/api/ai/analysis/result")
def analysis_result():
    return {
        "result": "success",
        "analysis": latest_analysis
    }