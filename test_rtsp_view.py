import cv2

url = "rtsp://192.168.1.3:8554/cam"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("RTSP 연결 실패")
    exit()

print("RTSP 연결 성공. q 누르면 종료")

while True:
    # 쌓인 프레임 일부 버리고 최신 프레임 쪽으로 이동
    for _ in range(3):
        cap.grab()

    ret, frame = cap.retrieve()

    if not ret:
        print("프레임 읽기 실패")
        continue

    frame = cv2.resize(frame, (640, 360))

    cv2.imshow("Raspberry Pi Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()