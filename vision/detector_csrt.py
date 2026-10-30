"""YOLO + CSRT/HSV tracker with target-conflict rejection and SOM continuity."""
import math
import time

import cv2
from ultralytics import YOLO

BOX_SCALE = 0.75
VERIFY_INTERVAL = 10.0
RECOVERY_INTERVAL = 1.0
COLOR_CHECK_EVERY = 3
COLOR_BAD_LIMIT = 4
COLOR_DISTANCE_LIMIT = 0.48
MAX_CENTER_JUMP = 2.5
TARGETS = ("goldfish", "fishmodel")

SAME_TARGET_MAX_GAP = 4.0
SAME_TARGET_MAX_JUMP = 2.0

# A goldfish candidate cannot replace the model fish at the same position
# unless its appearance matches the previously learned goldfish appearance.
MODEL_OVERLAP_LIMIT = 0.55
IDENTITY_COLOR_LIMIT = 0.42
CONFLICT_FRAMES = 2


def csrt_create():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise RuntimeError("Install opencv-contrib-python to use CSRT")


def bound_box(b, shape):
    H, W = shape[:2]
    x, y, w, h = map(float, b)
    x1, y1 = max(0, min(W, round(x))), max(0, min(H, round(y)))
    x2, y2 = max(0, min(W, round(x + w))), max(0, min(H, round(y + h)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2 - x1, y2 - y1


def yolo_to_csrt(xyxy, shape):
    x1, y1, x2, y2 = map(float, xyxy)
    w, h = (x2 - x1) * BOX_SCALE, (y2 - y1) * BOX_SCALE
    return bound_box(((x1 + x2 - w) / 2, (y1 + y2 - h) / 2, w, h), shape)


def center(box):
    x, y, w, h = box
    return int(x + w / 2), int(y + h / 2)


def overlap_ratio(a, b):
    """Intersection / smaller box area, useful when one box is smaller."""
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    h = max(0, min(ay + ah, by + bh) - max(ay, by))
    return (w * h) / max(1, min(aw * ah, bw * bh))


def histogram(frame, bbox):
    x, y, w, h = bbox
    x += int(w * 0.15)
    y += int(h * 0.15)
    w, h = int(w * 0.70), int(h * 0.70)
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0, cv2.NORM_L1)
    return hist


def hist_distance(a, b):
    if a is None or b is None:
        return 1.0
    return cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)


def detect_both(model, frame):
    results = model.predict(frame, conf=0.2, verbose=False)
    found = {name: [] for name in TARGETS}
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            name = str(model.names[int(box.cls[0].item())]).lower()
            if name not in found:
                continue
            bbox = yolo_to_csrt(box.xyxy[0].cpu().tolist(), frame.shape)
            if bbox is not None:
                found[name].append((bbox, float(box.conf[0].item())))
    return found


def pick_detection(detections, old_box, strict):
    if not detections:
        return None
    if old_box is None:
        return max(detections, key=lambda d: d[1])[0]
    previous = center(old_box)
    nearest, _ = min(detections, key=lambda d: math.dist(center(d[0]), previous))
    max_dist = max(old_box[2], old_box[3]) * MAX_CENTER_JUMP
    if strict and math.dist(center(nearest), previous) > max_dist:
        return None
    return nearest


def create_tracker(frame, bbox):
    tracker = csrt_create()
    tracker.init(frame, bbox)
    return tracker


class TrackState:
    def __init__(self, name):
        self.name = name
        self.generation = 0
        self.identity_hist = None  # Keep through short losses; clear on full reset
        self.reset()

    def reset(self, clear_identity=False):
        self.tracker = None
        self.bbox = None
        self.reference_hist = None
        self.color_failures = 0
        self.conflict_count = 0
        self.state = "WAITING"
        self.last_good_at = None
        if clear_identity:
            self.identity_hist = None
        self.generation += 1

    def update_csrt(self, frame, frame_number, now):
        if self.tracker is None:
            return False
        ok, tracked = self.tracker.update(frame)
        candidate = bound_box(tracked, frame.shape) if ok else None
        if candidate is None:
            self.state = "LOST"
            return True
        self.bbox = candidate
        self.state = "TRACKING"
        if frame_number % COLOR_CHECK_EVERY == 0 and self.reference_hist is not None:
            current = histogram(frame, self.bbox)
            dist = hist_distance(self.reference_hist, current)
            self.color_failures = self.color_failures + 1 if dist > COLOR_DISTANCE_LIMIT else 0
        if self.color_failures >= COLOR_BAD_LIMIT:
            self.state = "SUSPECT"
            return True
        self.last_good_at = now
        return False

    def is_same_target(self, new_box, now):
        if self.bbox is None or self.last_good_at is None:
            return False
        gap = now - self.last_good_at
        if not 0 <= gap <= SAME_TARGET_MAX_GAP:
            return False
        distance = math.dist(center(self.bbox), center(new_box))
        return distance <= max(self.bbox[2], self.bbox[3]) * SAME_TARGET_MAX_JUMP

    def verify(self, frame, found, suspect, now, model_box=None):
        # Reject goldfish detections sitting on top of the tracked model fish,
        # unless the appearance is consistent with the established goldfish.
        if self.name == "goldfish" and model_box is not None:
            safe = []
            for bbox, conf in found:
                if overlap_ratio(bbox, model_box) >= MODEL_OVERLAP_LIMIT:
                    appearance = hist_distance(self.identity_hist, histogram(frame, bbox))
                    if self.identity_hist is None or appearance > IDENTITY_COLOR_LIMIT:
                        print("[TRACK] reject goldfish YOLO box overlapping fishmodel", flush=True)
                        continue
                safe.append((bbox, conf))
            found = safe

        new_box = pick_detection(
            found, self.bbox,
            strict=(self.tracker is not None and not suspect)
        )
        if new_box is not None:
            need_reset = self.tracker is None or suspect
            if need_reset:
                same_target = self.is_same_target(new_box, now)
                new_hist = histogram(frame, new_box)
                # Don't learn the wrong model-fish colors as goldfish identity.
                if self.name == "goldfish" and self.identity_hist is not None:
                    if hist_distance(self.identity_hist, new_hist) > IDENTITY_COLOR_LIMIT:
                        same_target = False
                        if overlap_ratio(new_box, model_box) >= MODEL_OVERLAP_LIMIT:
                            return
                self.tracker = create_tracker(frame, new_box)
                self.bbox = new_box
                self.reference_hist = new_hist
                self.color_failures = 0
                self.conflict_count = 0
                self.state = "TRACKING"
                self.last_good_at = now
                if self.name == "goldfish" and self.identity_hist is None and new_hist is not None:
                    self.identity_hist = new_hist.copy()
                if same_target:
                    print(f"[YOLO] {self.name} CSRT recovered (SOM maintained)", flush=True)
                else:
                    self.generation += 1
                    print(f"[YOLO] {self.name} CSRT initialized/recovered (SOM reset)", flush=True)
            elif self.reference_hist is None:
                self.reference_hist = histogram(frame, self.bbox)
        elif suspect:
            self.reset()
            self.state = "LOST"
            print(f"[TRACK] {self.name} lost; waiting for YOLO", flush=True)


class FishCSRTTracker:
    def __init__(self, weights_path, yolo_gate):
        self.model = YOLO(weights_path)
        available = {str(v).lower() for v in self.model.names.values()}
        missing = set(TARGETS) - available
        if missing:
            raise RuntimeError(f"Tracking weights missing classes {sorted(missing)}; available={sorted(available)}")
        self.yolo_gate = yolo_gate
        self.targets = {name: TrackState(name) for name in TARGETS}
        self.frames = 0
        self.last_verify = 0.0
        self.last_attempt = 0.0

    def reset(self):
        for target in self.targets.values():
            target.reset(clear_identity=True)
        self.last_verify = 0.0
        self.last_attempt = 0.0

    def reject_model_drift(self, frame):
        gold = self.targets["goldfish"]
        model = self.targets["fishmodel"]
        if gold.state != "TRACKING" or model.state != "TRACKING":
            gold.conflict_count = 0
            return
        overlap = overlap_ratio(gold.bbox, model.bbox)
        if overlap < MODEL_OVERLAP_LIMIT:
            gold.conflict_count = 0
            return
        appearance = hist_distance(gold.identity_hist, histogram(frame, gold.bbox))
        if gold.identity_hist is not None and appearance <= IDENTITY_COLOR_LIMIT:
            gold.conflict_count = 0  # Goldfish may actually swim past the model
            return
        gold.conflict_count += 1
        if gold.conflict_count >= CONFLICT_FRAMES:
            gold.reset()  # forces SOM window reset, avoids false movement paths
            gold.state = "LOST"
            print("[TRACK] goldfish overlapped fishmodel; dropping wrong tracker", flush=True)

    def update(self, frame, now=None):
        now = time.monotonic() if now is None else now
        self.frames += 1
        suspects = {
            name: target.update_csrt(frame, self.frames, now)
            for name, target in self.targets.items()
        }
        self.reject_model_drift(frame)
        periodic = now - self.last_verify >= VERIFY_INTERVAL
        recovery = (
            any(t.tracker is None or suspects[name] for name, t in self.targets.items())
            and now - self.last_attempt >= RECOVERY_INTERVAL
        )
        if periodic or recovery:
            self.last_attempt = now
            try:
                with self.yolo_gate.high():
                    found = detect_both(self.model, frame)
            except Exception as exc:
                print(f"[YOLO] Error: {exc}", flush=True)
                found = {name: [] for name in TARGETS}
            self.last_verify = time.monotonic()
            model = self.targets["fishmodel"]
            model.verify(frame, found["fishmodel"], suspects["fishmodel"], now)
            model_box = model.bbox if model.state == "TRACKING" else None
            self.targets["goldfish"].verify(
                frame, found["goldfish"], suspects["goldfish"], now, model_box=model_box
            )
            self.reject_model_drift(frame)
        return {
            name: t.bbox if t.tracker is not None and t.state == "TRACKING" else None
            for name, t in self.targets.items()
        }
