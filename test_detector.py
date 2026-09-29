import cv2

from vision.detector import FishDetector


VIDEO_PATH = "goldfish.mp4"


# 영상 열기
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("영상 열기 실패")
    exit()


# YOLO + ByteTrack
detector = FishDetector()


# 결과 창
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("result", 640, 360)


print("YOLO + ByteTrack 테스트 시작 - q 누르면 종료")


while True:
    ret, frame = cap.read()

    if not ret:
        print("영상 종료")
        break


    # YOLO + ByteTrack 실행
    results = detector.detect(frame)

    result = results[0]
    boxes = result.boxes


    # =========================
    # 추적 결과 출력
    # =========================

    if boxes is not None and boxes.id is not None:

        for i in range(len(boxes)):

            track_id = int(boxes.id[i].item())
            class_id = int(boxes.cls[i].item())
            class_name = detector.model.names[class_id]
            confidence = float(boxes.conf[i].item())

            print({
                "trackId": track_id,
                "className": class_name,
                "confidence": round(confidence, 4)
            })


    # =========================
    # 화면 표시
    # =========================

    annotated = result.plot()

    cv2.imshow(
        "result",
        annotated
    )


    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # X 버튼으로 종료
    if cv2.getWindowProperty(
        "result",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break


cap.release()
cv2.destroyAllWindows()