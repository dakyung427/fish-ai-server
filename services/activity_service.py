from datetime import datetime

import cv2

import state

from ai_models import activity_model


def rtsp_loop(rtsp_url):

    # RTSP 연결
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():

        print("[RTSP] 연결 실패")

        state.is_running = False

        cap.release()

        return

    print("[RTSP] 연결 성공")

    frame_count = 0

    # 분석이 실행 중인 동안 계속 반복
    while state.is_running:

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

            # =====================================
            # 추후 구현
            #
            # YOLO Detection
            # ↓
            # Tracking
            # ↓
            # 행동 특징 추출
            # ↓
            # SOM 입력
            # ↓
            # anomalyScore 계산
            # ↓
            # 정상 / 이상 판단
            # =====================================

            state.activity_result = {
                "activityStatus": "정상",
                "anomalyScore": 0.1,
                "updatedAt": datetime.now().isoformat(
                    timespec="seconds"
                )
            }

    cap.release()

    print("[RTSP] 분석 종료")