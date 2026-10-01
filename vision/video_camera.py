import cv2


class VideoCamera:
    def __init__(self, video_path):
        self.video_path = video_path

        self.cap = cv2.VideoCapture(
            self.video_path
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"테스트 영상 열기 실패: {self.video_path}"
            )

        print(
            f"테스트 영상 연결 성공: {self.video_path}"
        )

    # RTSPCamera와 사용 형태를 맞추기 위한 함수
    def start(self):
        pass

    # ==========================================
    # 현재 프레임 가져오기
    # ==========================================
    def read(self):

        ret, frame = self.cap.read()

        # 영상 끝까지 갔으면 처음부터 다시 재생
        if not ret:

            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            ret, frame = self.cap.read()

            if not ret:
                return None

        return frame

    # ==========================================
    # 영상 종료
    # ==========================================
    def stop(self):

        if self.cap is not None:
            self.cap.release()

        print("테스트 영상 종료")