
"""YOLO + CSRT/HSV tracker with SOM continuity-aware recovery."""

import math
import time

import cv2
from ultralytics import YOLO


# ==========================================
# 추적 설정
# ==========================================
BOX_SCALE = 0.75

VERIFY_INTERVAL = 10.0
RECOVERY_INTERVAL = 1.0

COLOR_CHECK_EVERY = 3
COLOR_BAD_LIMIT = 4
COLOR_DISTANCE_LIMIT = 0.48

MAX_CENTER_JUMP = 2.5

TARGETS = ("goldfish", "fishmodel")


# ==========================================
# SOM 연속성 유지 설정
# ==========================================

# 같은 물고기로 판단할 최대 복구 시간
SAME_TARGET_MAX_GAP = 2.0

# 이전 박스 크기 대비 최대 중심 이동 거리
SAME_TARGET_MAX_JUMP = 1.5


def csrt_create():

    if (
        hasattr(cv2, "legacy")
        and hasattr(cv2.legacy, "TrackerCSRT_create")
    ):
        return cv2.legacy.TrackerCSRT_create()

    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    raise RuntimeError(
        "Install opencv-contrib-python to use CSRT"
    )


def bound_box(b, shape):

    H, W = shape[:2]

    x, y, w, h = map(float, b)

    x1 = max(0, min(W, round(x)))
    y1 = max(0, min(H, round(y)))

    x2 = max(0, min(W, round(x + w)))
    y2 = max(0, min(H, round(y + h)))

    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1
    )


def yolo_to_csrt(xyxy, shape):

    x1, y1, x2, y2 = map(float, xyxy)

    w = (x2 - x1) * BOX_SCALE
    h = (y2 - y1) * BOX_SCALE

    return bound_box(
        (
            (x1 + x2 - w) / 2,
            (y1 + y2 - h) / 2,
            w,
            h
        ),
        shape
    )


def center(b):

    x, y, w, h = b

    return (
        int(x + w / 2),
        int(y + h / 2)
    )


# ==========================================
# HSV 색상 히스토그램
# ==========================================
def histogram(frame, bbox):

    x, y, w, h = bbox

    x += int(w * 0.15)
    y += int(h * 0.15)

    w = int(w * 0.70)
    h = int(h * 0.70)

    crop = frame[y:y + h, x:x + w]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [24, 16],
        [0, 180, 0, 256]
    )

    cv2.normalize(
        hist,
        hist,
        1.0,
        0,
        cv2.NORM_L1
    )

    return hist


# ==========================================
# YOLO 검출
# ==========================================
def detect_both(model, frame):

    results = model.predict(
        frame,
        conf=0.2,
        verbose=False
    )

    found = {
        name: []
        for name in TARGETS
    }

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            name = str(
                model.names[
                    int(box.cls[0].item())
                ]
            ).lower()

            if name not in found:
                continue

            bbox = yolo_to_csrt(
                box.xyxy[0].cpu().tolist(),
                frame.shape
            )

            if bbox is not None:

                found[name].append((
                    bbox,
                    float(box.conf[0].item())
                ))

    return found


# ==========================================
# 가장 적합한 YOLO 검출 선택
# ==========================================
def pick_detection(detections, old_box, strict):

    if not detections:
        return None

    if old_box is None:
        return max(
            detections,
            key=lambda d: d[1]
        )[0]

    previous = center(old_box)

    scored = sorted(
        detections,
        key=lambda d: math.dist(
            center(d[0]),
            previous
        )
    )

    nearest, _conf = scored[0]

    max_dist = (
        max(old_box[2], old_box[3])
        * MAX_CENTER_JUMP
    )

    if (
        strict
        and math.dist(
            center(nearest),
            previous
        ) > max_dist
    ):
        return None

    return nearest


# ==========================================
# CSRT 생성
# ==========================================
def create_tracker(frame, bbox):

    tr = csrt_create()
    tr.init(frame, bbox)

    return tr


class TrackState:

    def __init__(self, name):

        self.name = name
        self.generation = 0
        self.reset()

    # ==========================================
    # 추적 상태 초기화
    # ==========================================
    def reset(self):

        self.tracker = None
        self.bbox = None

        self.reference_hist = None
        self.color_failures = 0

        self.state = "WAITING"

        # 마지막 정상 추적 시각
        self.last_good_at = None

        # 실제 추적 연속성 초기화
        self.generation += 1

    # ==========================================
    # CSRT 추적
    # ==========================================
    def update_csrt(self, frame, frame_number, now):

        if self.tracker is None:
            return False

        ok, tracked = self.tracker.update(frame)

        candidate = (
            bound_box(tracked, frame.shape)
            if ok
            else None
        )

        if candidate is None:
            self.state = "LOST"
            return True

        self.bbox = candidate
        self.state = "TRACKING"

        # ======================================
        # HSV 색상 검증
        # ======================================
        if (
            frame_number % COLOR_CHECK_EVERY == 0
            and self.reference_hist is not None
        ):

            current = histogram(
                frame,
                self.bbox
            )

            dist = (
                cv2.compareHist(
                    self.reference_hist,
                    current,
                    cv2.HISTCMP_BHATTACHARYYA
                )
                if current is not None
                else 1.0
            )

            if dist > COLOR_DISTANCE_LIMIT:
                self.color_failures += 1
            else:
                self.color_failures = 0

        if self.color_failures >= COLOR_BAD_LIMIT:

            self.state = "SUSPECT"
            return True

        # 정상 추적된 마지막 시각 기록
        self.last_good_at = now

        return False

    # ==========================================
    # 같은 개체인지 연속성 확인
    # ==========================================
    def is_same_target(self, new_box, now):

        if self.bbox is None:
            return False

        if self.last_good_at is None:
            return False

        # 마지막 정상 추적 이후 경과시간
        gap = now - self.last_good_at

        if (
            gap < 0
            or gap > SAME_TARGET_MAX_GAP
        ):
            return False

        # 이전 박스와 새로운 박스의 중심 거리
        distance = math.dist(
            center(self.bbox),
            center(new_box)
        )

        max_distance = (
            max(
                self.bbox[2],
                self.bbox[3]
            )
            * SAME_TARGET_MAX_JUMP
        )

        if distance > max_distance:
            return False

        return True

    # ==========================================
    # YOLO 검증 및 CSRT 복구
    # ==========================================
    def verify(self, frame, found, suspect, now):

        new_box = pick_detection(
            found,
            self.bbox,
            strict=(
                self.tracker is not None
                and not suspect
            )
        )

        if new_box is not None:

            need_reset = (
                self.tracker is None
                or suspect
            )

            if need_reset:

                # 이전 개체와 동일한지 먼저 확인
                same_target = self.is_same_target(
                    new_box,
                    now
                )

                # ==================================
                # CSRT 추적기 재생성
                # ==================================
                self.tracker = create_tracker(
                    frame,
                    new_box
                )

                self.bbox = new_box

                self.reference_hist = histogram(
                    frame,
                    new_box
                )

                self.color_failures = 0
                self.state = "TRACKING"
                self.last_good_at = now

                if same_target:

                    # 가까운 위치에서 빠르게 복구됨
                    # SOM 분석 윈도우 유지
                    print(
                        f"[YOLO] {self.name} CSRT recovered "
                        f"(SOM maintained)",
                        flush=True
                    )

                else:

                    # 새로운 추적 시작 또는 연속성 실패
                    # 기존 SOM 데이터 초기화
                    self.generation += 1

                    print(
                        f"[YOLO] {self.name} CSRT initialized/recovered "
                        f"(SOM reset)",
                        flush=True
                    )

            elif self.reference_hist is None:

                self.reference_hist = histogram(
                    frame,
                    self.bbox
                )

        elif suspect:

            # 실제 추적 실패
            self.reset()
            self.state = "LOST"

            print(
                f"[TRACK] {self.name} lost; "
                f"waiting for YOLO",
                flush=True
            )


class FishCSRTTracker:

    def __init__(self, weights_path, yolo_gate):

        self.model = YOLO(weights_path)

        available = {
            str(v).lower()
            for v in self.model.names.values()
        }

        missing = set(TARGETS) - available

        if missing:

            raise RuntimeError(
                f"Tracking weights missing classes "
                f"{sorted(missing)}; "
                f"available={sorted(available)}"
            )

        self.yolo_gate = yolo_gate

        self.targets = {
            name: TrackState(name)
            for name in TARGETS
        }

        self.frames = 0
        self.last_verify = 0.0
        self.last_attempt = 0.0

    # ==========================================
    # 전체 추적 초기화
    # ==========================================
    def reset(self):

        for target in self.targets.values():
            target.reset()

        self.last_verify = 0.0
        self.last_attempt = 0.0

    # ==========================================
    # 실시간 YOLO + CSRT
    # ==========================================
    def update(self, frame, now=None):

        now = (
            time.monotonic()
            if now is None
            else now
        )

        self.frames += 1

        suspects = {
            name: target.update_csrt(
                frame,
                self.frames,
                now
            )
            for name, target in self.targets.items()
        }

        # 10초 주기 YOLO 확인
        periodic = (
            now - self.last_verify
            >= VERIFY_INTERVAL
        )

        # 추적 실패 시 빠른 복구
        recovery = (
            any(
                t.tracker is None or suspects[name]
                for name, t in self.targets.items()
            )
            and now - self.last_attempt
            >= RECOVERY_INTERVAL
        )

        if periodic or recovery:

            self.last_attempt = now

            try:

                # 다른 YOLO 작업과 동시 실행 방지
                with self.yolo_gate.high():

                    found = detect_both(
                        self.model,
                        frame
                    )

            except Exception as exc:

                print(
                    f"[YOLO] Error: {exc}",
                    flush=True
                )

                found = {
                    name: []
                    for name in TARGETS
                }

            self.last_verify = time.monotonic()

            for name, target in self.targets.items():

                target.verify(
                    frame,
                    found[name],
                    suspects[name],
                    now
                )

        return {
            name: (
                t.bbox
                if (
                    t.tracker is not None
                    and t.state == "TRACKING"
                )
                else None
            )
            for name, t in self.targets.items()
        }
