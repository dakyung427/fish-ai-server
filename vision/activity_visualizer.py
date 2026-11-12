
"""Render existing CSRT/SOM results; never runs its own inference or camera."""

import threading
from collections import deque

import cv2


class ActivityVisualizer:
    def __init__(self, max_trail=100, jpeg_quality=80):
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.trails = {}
        self.generations = {}
        self.max_trail = max_trail
        self.jpeg_quality = jpeg_quality

    def get_frame(self):
        with self.lock:
            return self.latest_jpeg

    def clear(self):
        with self.lock:
            self.latest_jpeg = None
        self.trails.clear()

    def update(self, frame, boxes, analyzers, targets):
        display = frame.copy()

        for idx, (name, target) in enumerate(targets.items()):
            analyzer = analyzers[name]
            bbox = boxes.get(name)

            # 추적 세대 변경 시 이동 경로 초기화
            if self.generations.get(name) != target.generation:
                self.trails[name] = deque(maxlen=self.max_trail)
                self.generations[name] = target.generation

            trail = self.trails[name]

            # SOM 최종 상태에 따른 색상
            color = (
                (0, 0, 255)
                if analyzer.final_abnormal
                else (0, 255, 0)
            )

            if bbox is not None and target.state == "TRACKING":
                x, y, w, h = map(int, bbox)
                point = (x + w // 2, y + h // 2)

                # 이동 경로 기록
                trail.append(point)

                # CSRT 추적 박스
                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

                # 중심점
                cv2.circle(
                    display,
                    point,
                    3,
                    (0, 0, 255),
                    -1
                )

                # 이동 경로
                points = list(trail)

                for a, b in zip(points[:-1], points[1:]):
                    cv2.line(
                        display,
                        a,
                        b,
                        color,
                        1
                    )

                # =====================================
                # SOM 판정 결과 및 점수 표시
                # =====================================
                if analyzer.last_result in ("NORMAL", "WARNING"):

                    status_text = (
                        "WARNING"
                        if analyzer.final_abnormal
                        else "NORMAL"
                    )

                    y = 32 + idx * 65

                    # NORMAL / WARNING
                    cv2.putText(
                        display,
                        status_text,
                        (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                        cv2.LINE_AA
                    )

                    # SOM 점수 / 임계값
                    if analyzer.last_score is not None:
                        cv2.putText(
                            display,
                            f"SOM Score: {analyzer.last_score:.3f}/{analyzer.threshold:.3f}",
                            (12, y + 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

            else:
                trail.clear()
                continue

        # JPEG 인코딩
        ok, encoded = cv2.imencode(
            ".jpg",
            display,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )

        if ok:
            with self.lock:
                self.latest_jpeg = encoded.tobytes()
