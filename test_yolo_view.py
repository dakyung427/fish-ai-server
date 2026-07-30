from ultralytics import YOLO
import cv2

url = "rtsp://192.168.1.3:8554/cam"

model = YOLO("best.pt")

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("RTSP 연결 실패")
    exit()

print("RTSP 연결 성공. q 누르면 종료")

frame_count = 0
last_boxes = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임 읽기 실패")
        break

    frame_count += 1

    # YOLO는 15프레임마다 한 번만 실행
    if frame_count % 15 == 0:
        results = model(frame, conf=0.5, verbose=False)

        last_boxes = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            last_boxes.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "className": class_name,
                "confidence": confidence
            })

    # 매 프레임마다 마지막 박스를 현재 화면에 그림
    display_frame = frame.copy()

    for item in last_boxes:
        x1 = item["x1"]
        y1 = item["y1"]
        x2 = item["x2"]
        y2 = item["y2"]
        class_name = item["className"]
        confidence = item["confidence"]

        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            display_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Raspberry Pi Camera YOLO", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()