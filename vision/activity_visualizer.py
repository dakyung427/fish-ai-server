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
            if self.generations.get(name) != target.generation:
                self.trails[name] = deque(maxlen=self.max_trail)
                self.generations[name] = target.generation
            trail = self.trails[name]
            color = (0, 0, 255) if analyzer.final_abnormal else (0, 255, 0)
            if bbox is not None and target.state == 'TRACKING':
                x, y, w, h = map(int, bbox)
                point = (x + w // 2, y + h // 2)
                trail.append(point)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 1)
                cv2.circle(display, point, 3, (0, 0, 255), -1)
                points = list(trail)
                for a, b in zip(points[:-1], points[1:]):
                    cv2.line(display, a, b, color, 1)
                cv2.putText(display, name.upper(), (x, max(15, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            else:
                trail.clear()
            y = 26 + idx * 84
            cv2.putText(display, f'{name.upper()} {target.state}', (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1)
            cv2.putText(display, f'SOM: {analyzer.last_result} | WARNING: {analyzer.warning}',
                        (12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1)
            if analyzer.last_score is not None:
                cv2.putText(display,
                            f'Score {analyzer.last_score:.3f}/{analyzer.threshold:.3f}',
                            (12, y + 48), cv2.FONT_HERSHEY_SIMPLEX, .48,
                            (255, 255, 255), 1)
        ok, encoded = cv2.imencode('.jpg', display,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            with self.lock:
                self.latest_jpeg = encoded.tobytes()
