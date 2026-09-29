import cv2
import threading
import time

from config import RTSP_URL


class RTSPCamera:
    def __init__(self, url=RTSP_URL):
        self.url = url
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

        self.connect()

    # RTSP 연결
    def connect(self):
        print("RTSP 연결 시도...")

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(
            self.url,
            cv2.CAP_FFMPEG
        )

        # 영상 지연을 줄이기 위해 버퍼 최소화
        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        if self.cap.isOpened():
            print("RTSP 연결 성공")
            return True

        print("RTSP 연결 실패")
        return False

    # RTSP 수신 스레드 시작
    def start(self):
        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )

        self.thread.start()

    # RTSP를 계속 읽고 가장 최신 프레임만 저장
    def update(self):
        while self.running:

            # 연결이 끊긴 경우 재연결
            if self.cap is None or not self.cap.isOpened():

                time.sleep(1)
                self.connect()

                continue

            ret, frame = self.cap.read()

            # 프레임 읽기 실패 시 재연결
            if not ret:

                print("프레임 읽기 실패 → RTSP 재연결")

                self.cap.release()

                time.sleep(1)

                self.connect()

                continue

            # 이전 프레임은 버리고
            # 최신 프레임으로 계속 덮어쓰기
            with self.lock:
                self.frame = frame

    # 가장 최신 프레임 가져오기
    def read(self):
        with self.lock:
            return self.frame

    # RTSP 종료
    def stop(self):

        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2)

        if self.cap is not None:
            self.cap.release()