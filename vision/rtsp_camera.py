
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

        # 새 프레임 확인용
        self.sequence = 0
        self.last_frame_at = 0.0
        self.connected = False

        self.connect()

    # ==========================================
    # RTSP 연결
    # ==========================================
    def connect(self):
        print("RTSP 연결 시도...")

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(
            self.url,
            cv2.CAP_FFMPEG
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        with self.lock:
            self.connected = self.cap.isOpened()
            self.frame = None

        if self.cap.isOpened():
            print("RTSP 연결 성공")
            return True

        print("RTSP 연결 실패")
        return False

    # ==========================================
    # RTSP 수신 스레드
    # ==========================================
    def start(self):
        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )

        self.thread.start()

    def update(self):

        while self.running:

            if self.cap is None or not self.cap.isOpened():

                time.sleep(1)

                if self.running:
                    self.connect()

                continue

            ret, frame = self.cap.read()

            if not ret:

                print("프레임 읽기 실패 → RTSP 재연결")

                with self.lock:
                    self.connected = False
                    self.frame = None

                self.cap.release()
                self.cap = None

                time.sleep(1)

                if self.running:
                    self.connect()

                continue

            # 최신 프레임 + 시각 + 번호 기록
            with self.lock:
                self.frame = frame
                self.sequence += 1
                self.last_frame_at = time.monotonic()
                self.connected = True

    # ==========================================
    # 기존 어종 분석용
    # ==========================================
    def read(self):
        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()

    # ==========================================
    # 활동성 분석용
    # ==========================================
    def read_latest(self):
        with self.lock:
            frame = (
                None if self.frame is None
                else self.frame.copy()
            )

            return (
                frame,
                self.sequence,
                self.last_frame_at,
                self.connected
            )

    # ==========================================
    # RTSP 종료
    # ==========================================
    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        with self.lock:
            self.connected = False
            self.frame = None
