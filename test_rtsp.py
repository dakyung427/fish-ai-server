import cv2
import time

from vision.rtsp_camera import RTSPCamera


# RTSP 카메라 생성
camera = RTSPCamera()

# 영상 수신 시작
camera.start()

print("실시간 RTSP 시작 - q 누르면 종료")


while True:

    # 가장 최신 프레임 가져오기
    frame = camera.read()

    # 아직 프레임이 없으면 잠시 대기
    if frame is None:
        time.sleep(0.01)
        continue

    # 영상 화면 출력
    cv2.imshow(
        "Raspberry Pi Camera",
        frame
    )

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# 종료 처리
camera.stop()

cv2.destroyAllWindows()