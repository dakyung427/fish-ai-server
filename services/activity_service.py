"""10-second SOM activity analysis from a shared, sequence-numbered RTSP camera."""

import math
import threading
import time

import numpy as np

from vision.activity_visualizer import ActivityVisualizer
from vision.detector_csrt import center, histogram, hist_distance


# ==========================================
# SOM 분석 설정
# ==========================================
WINDOW_SECONDS = 10.0
JITTER_DEADBAND_BODY = 0.030
SOM_UPDATE_SECONDS = 3.0
MAX_RESULT_HOLD = 15.0
RECOVERY_PROBE_SECONDS = 0.8
MAX_TRACKING_GAP = 0.8
SMOOTH_RADIUS = 3


class SOMAnalyzer:

    def __init__(self, model_path):

        with np.load(model_path, allow_pickle=False) as m:
            self.weights = m["weights"]
            self.mean = m["mean"]
            self.scale = m["scale"]
            self.threshold = float(m["threshold"])
            self.features = [str(v) for v in m["features"]]

        expected = [
            "distance_body",
            "speed_body_s",
            "still_ratio",
            "net_displacement_body",
        ]

        if self.features != expected:
            raise ValueError(
                f"Unexpected SOM features: {self.features}"
            )

        self.samples = []
        self.start = None
        self.next_eval_t = None

        self.warning = 0
        self.final_abnormal = False

        self.last_score = None
        self.last_features = None
        self.last_result = "WAITING"

        # 마지막 정상 SOM 분석 완료 시각
        self.last_analyzed_at = None

    # ==========================================
    # 추적 변경 시 윈도우 초기화 (이전 표시 결과는 잠깐 HOLD)
    # ==========================================
    def reset(self, target_name="UNKNOWN", reason="tracking reset"):
        print(f"[SOM RESET][{target_name.upper()}] reason={reason} "
              f"samples={len(self.samples)} warning={self.warning}", flush=True)
        self.samples.clear()
        self.start = None
        self.next_eval_t = None
        self.warning = 0
        self.final_abnormal = False
        self.last_result = "PAUSED" if self.last_score is not None else "WAITING"
        # last_score/last_features/last_analyzed_at: 과거값 표시용. 새 판정이 아님.

    def add(self, timestamp, bbox, target_name="UNKNOWN"):
        # 마지막 유효 결과의 표시 기한 만료
        if (self.last_result in ("PAUSED", "NORMAL", "WARNING", "INSUFFICIENT TRACKING")
                and self.last_analyzed_at is not None
                and time.monotonic() - self.last_analyzed_at > MAX_RESULT_HOLD):
            self.last_result = "WAITING"
            self.last_score = None
            self.last_features = None

        if bbox is None:
            sample = (timestamp, float("nan"), float("nan"), float("nan"), 0)
        else:
            bx, by, bw, bh = bbox
            sample = (timestamp, bx + bw / 2, by + bh / 2, math.hypot(bw, bh), 1)

        self.samples.append(sample)
        cutoff = timestamp - WINDOW_SECONDS
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.pop(0)

        if self.next_eval_t is None:
            self.next_eval_t = timestamp + WINDOW_SECONDS
            return
        if timestamp + 1e-9 < self.next_eval_t:
            return
        self.next_eval_t = timestamp + SOM_UPDATE_SECONDS

        # 정확한 10초 구간이 없으면 분석하지 않음
        if not self.samples or timestamp - self.samples[0][0] < WINDOW_SECONDS - 0.25:
            return

        row = self.features_for_window(self.samples)
        if row is None:
            if self.last_result not in ("PAUSED", "NORMAL", "WARNING"):
                self.last_result = "INSUFFICIENT TRACKING"
            print(f"[SOM SKIP][{target_name.upper()}] insufficient 10s tracking", flush=True)
            return

        # ======================================
        # SOM 이상 점수 계산
        # ======================================
        raw = np.array(
            [row[name] for name in self.features],
            dtype=float,
        )

        z = np.clip(
            (raw - self.mean) / self.scale,
            -8,
            8,
        )

        score = float(
            np.sqrt(
                np.min(
                    np.sum(
                        (self.weights - z) ** 2,
                        axis=2,
                    )
                )
            )
        )

        abnormal = score > self.threshold

        # 이상 점수는 경고 누적 +1 (최대 2).
        # 정상 점수라도 정지 비율이 60% 이상이면 누적 경고를 유지한다.
        # WARNING 진입은 누적 2, 해제는 누적 0으로 히스테리시스 적용.
        still_ratio = row["still_ratio"]
        if abnormal:
            self.warning = min(2, self.warning + 1)
        elif still_ratio < 0.60:
            self.warning = max(0, self.warning - 1)

        if self.warning >= 2:
            self.final_abnormal = True
        elif self.warning == 0:
            self.final_abnormal = False
        # warning == 1이면 직전 확정 상태 유지

        self.last_score = score
        self.last_features = row

        self.last_result = (
            "WARNING"
            if self.final_abnormal
            else "NORMAL"
        )

        # 실제 SOM 분석이 완료된 시각
        self.last_analyzed_at = time.monotonic()

        print(
            f"[SOM][{target_name.upper()}] "
            f"score={score:.4f} "
            f"threshold={self.threshold:.4f} "
            f"still={row['still_ratio']:.2%} "
            f"distance={row['distance_body']:.3f} "
            f"window={'ABNORMAL' if abnormal else 'NORMAL'} "
            f"WARNING={self.warning} "
            f"status={self.last_result}",
            flush=True,
        )

    # ==========================================
    # 10초 이동 특징 추출
    # ==========================================
    @staticmethod
    def features_for_window(samples):

        if not samples:
            return None

        valid = [s for s in samples if s[4]]

        coverage = len(valid) / len(samples)

        if coverage < 0.8 or len(valid) < 5:
            return None

        times = np.array([s[0] for s in valid])

        xy = np.array(
            [[s[1], s[2]] for s in valid],
            dtype=float,
        )

        body = np.array(
            [s[3] for s in valid],
            dtype=float,
        )

        # 좌표 스무딩
        smooth = xy.copy()

        for i in range(len(xy)):
            smooth[i] = np.median(
                xy[
                    max(0, i - SMOOTH_RADIUS):
                    min(len(xy), i + SMOOTH_RADIUS + 1)
                ],
                axis=0,
            )

        intervals = np.diff(times)
        positive = intervals[intervals > 0]

        if not len(positive):
            return None

        expected_dt = float(np.median(positive))

        distance = 0.0
        still_dt = 0.0
        observed_dt = 0.0

        for i in range(1, len(valid)):

            dt = times[i] - times[i - 1]

            if dt <= 0 or dt > max(0.7, expected_dt * 2.5):
                continue

            unit = max(
                1.0,
                (body[i] + body[i - 1]) / 2,
            )

            d = math.dist(
                smooth[i],
                smooth[i - 1],
            ) / unit

            if d < JITTER_DEADBAND_BODY:
                d = 0.0

            distance += d
            observed_dt += dt

            if d / dt < 0.035:
                still_dt += dt

        if observed_dt < 5.0:
            return None

        net = math.dist(
            smooth[-1],
            smooth[0],
        ) / max(1.0, float(np.median(body)))

        if net < JITTER_DEADBAND_BODY:
            net = 0.0

        return {
            "distance_body": distance,
            "speed_body_s": distance / observed_dt,
            "still_ratio": still_dt / observed_dt,
            "net_displacement_body": net,
        }


class SOMContinuityGuard:
    """CSRT의 raw 박스와 SOM에 입력할 신뢰 박스를 분리한다 (v5 방식)."""

    def __init__(self, name):
        self.name = name
        self.clear()

    def clear(self):
        self.anchor = None  # (timestamp, box, HSV hist)
        self.suspect_since = None
        self.suspect_reason = None
        self.missing_since = None

    @staticmethod
    def _compare(anchor, box, frame, timestamp):
        old_t, old_box, old_hist = anchor
        old_size = max(1.0, math.hypot(old_box[2], old_box[3]))
        new_size = max(1.0, math.hypot(box[2], box[3]))
        ratio = max(old_size / new_size, new_size / old_size)
        jump = math.dist(center(old_box), center(box)) / old_size
        color = hist_distance(old_hist, histogram(frame, box))
        return timestamp - old_t, jump, color, ratio

    def accept(self, frame, timestamp, bbox, generation_changed, analyzer):
        """유효 SOM bbox 반환. None이면 그 순간의 좌표는 사용하지 않는다."""
        if bbox is None:
            if self.missing_since is None:
                self.missing_since = timestamp
            if (timestamp - self.missing_since > MAX_TRACKING_GAP
                    and self.anchor is not None):
                analyzer.reset(self.name, 'tracking missing > 0.8s')
                self.clear()
                self.missing_since = timestamp
            return None

        if self.anchor is None:
            self.anchor = (timestamp, tuple(bbox), histogram(frame, bbox))
            self.missing_since = None
            return bbox

        # 의심 상태: 최대 0.8초 신뢰할 수 있는 기존 박스로 돌아오는지 확인
        if self.suspect_since is not None:
            gap, jump, color, ratio = self._compare(self.anchor, bbox, frame, timestamp)
            recovered = (0 <= gap <= RECOVERY_PROBE_SECONDS + 0.25
                         and jump <= 0.65 and color <= 0.70 and ratio <= 1.6)
            if recovered:
                print(f"[SOM RECOVERED][{self.name.upper()}] t={timestamp:.1f} "
                      f"jump={jump:.2f} color={color:.2f}", flush=True)
                self.suspect_since = None
                self.suspect_reason = None
            elif timestamp - self.suspect_since >= RECOVERY_PROBE_SECONDS:
                analyzer.reset(self.name, f'probe timeout: {self.suspect_reason}')
                self.clear()
                self.anchor = (timestamp, tuple(bbox), histogram(frame, bbox))
                return bbox  # 새 트랙 첫 유효 샘플
            else:
                return None

        # generation 변경은 단독 RESET 사유가 아님: 실제 좌표 연속성을 검증
        gap, jump, color, ratio = self._compare(self.anchor, bbox, frame, timestamp)
        if generation_changed:
            strict = jump <= 0.75 and color <= 0.42
            near = gap <= 0.35 and ((jump <= 0.35 and color <= 0.72)
                                    or (jump <= 0.12 and color <= 0.80))
            moderate = gap <= 0.20 and jump <= 0.55 and color <= 0.60
            accepted = 0 <= gap <= 0.8 and ratio <= 1.8 and (strict or near or moderate)
            if not accepted:
                self.suspect_since = timestamp
                self.suspect_reason = (f'generation jump={jump:.2f} '
                                       f'color={color:.2f} size={ratio:.2f}')
                print(f"[SOM PROBE][{self.name.upper()}] t={timestamp:.1f} "
                      f"{self.suspect_reason}", flush=True)
                return None
            print(f"[SOM KEEP][{self.name.upper()}] generation changed "
                  f"jump={jump:.2f} color={color:.2f}", flush=True)
        elif 0 <= gap < 0.35 and jump > 1.25:
            self.suspect_since = timestamp
            self.suspect_reason = f'raw CSRT jump={jump:.2f}'
            print(f"[SOM PROBE][{self.name.upper()}] t={timestamp:.1f} "
                  f"{self.suspect_reason}", flush=True)
            return None

        # 큰 추적 공백을 건너뛴 좌표를 잇지 않기
        if self.missing_since is not None and timestamp - self.missing_since > MAX_TRACKING_GAP:
            analyzer.reset(self.name, 'recovered after long tracking loss')
        self.missing_since = None
        self.anchor = (timestamp, tuple(bbox), histogram(frame, bbox))
        return bbox


class ActivityService:

    def __init__(self, tracker, camera, model_path):

        self.tracker = tracker
        self.camera = camera

        self.analyzers = {
            name: SOMAnalyzer(model_path)
            for name in tracker.targets
        }

        self.stop_event = threading.Event()
        self.results_lock = threading.Lock()

        self.last_results = {}

        self.last_generations = {
            name: state.generation
            for name, state in tracker.targets.items()
        }

        self.guards = {name: SOMContinuityGuard(name) for name in tracker.targets}
        self.visualizer = ActivityVisualizer()

    # ==========================================
    # 종료
    # ==========================================
    def stop(self):
        self.stop_event.set()

    # ==========================================
    # 결과 스냅샷
    # ==========================================
    def snapshot(self):

        with self.results_lock:
            return {
                name: {
                    **data,
                    "features": (
                        data["features"].copy()
                        if data["features"] is not None
                        else None
                    ),
                }
                for name, data in self.last_results.items()
            }

    # ==========================================
    # 시각화 프레임
    # ==========================================
    def preview_snapshot(self):
        return self.visualizer.get_frame()

    # ==========================================
    # 현재 결과 업데이트
    # ==========================================
    def update_results(self):

        results = {}

        for name, analyzer in self.analyzers.items():

            results[name] = {
                "state": self.tracker.targets[name].state,
                "status": analyzer.last_result,
                "warning": analyzer.warning,
                "abnormal": analyzer.final_abnormal,
                "score": analyzer.last_score,
                "features": (
                    analyzer.last_features.copy()
                    if analyzer.last_features is not None
                    else None
                ),
                "last_analyzed_at": analyzer.last_analyzed_at,
            }

        with self.results_lock:
            self.last_results = results

    # ==========================================
    # 전체 추적 초기화
    # ==========================================
    def reset_tracking(self, reason="RTSP reset"):

        print(
            f"[TRACK RESET][ALL] reason={reason}",
            flush=True,
        )

        self.tracker.reset()

        for name, analyzer in self.analyzers.items():
            analyzer.reset(
                target_name=name,
                reason=reason,
            )

        self.last_generations = {
            name: state.generation
            for name, state in self.tracker.targets.items()
        }

        for guard in self.guards.values():
            guard.clear()
        self.visualizer.clear()
        self.update_results()

    # ==========================================
    # 실시간 YOLO + CSRT + SOM
    # ==========================================
    def run_sync(self):

        print(
            "[ACTIVITY] CSRT + SOM started",
            flush=True,
        )

        last_seq = -1
        last_sample = 0.0
        last_received_at = None
        stream_lost = False

        while not self.stop_event.is_set():

            now = time.monotonic()

            if now - last_sample < 0.10:
                self.stop_event.wait(0.01)
                continue

            frame, seq, received_at, connected = (
                self.camera.read_latest()
            )

            stale = (
                frame is None
                or not connected
                or now - received_at > 2.0
            )

            # ======================================
            # RTSP 연결 중단
            # ======================================
            if stale:

                if not stream_lost:

                    print(
                        "[RTSP] Stream unavailable; "
                        "resetting trackers",
                        flush=True,
                    )

                    self.reset_tracking(
                        reason="RTSP stream unavailable"
                    )

                    stream_lost = True

                self.stop_event.wait(0.05)
                continue

            if seq == last_seq:
                self.stop_event.wait(0.01)
                continue

            # ======================================
            # RTSP 연결 복구
            # ======================================
            if stream_lost:

                print(
                    "[RTSP] Stream recovered",
                    flush=True,
                )

                stream_lost = False
                last_received_at = None

            # ======================================
            # 프레임 간격 이상
            # ======================================
            if (
                last_received_at is not None
                and received_at - last_received_at > 2.0
            ):

                gap = received_at - last_received_at

                print(
                    f"[RTSP] Large frame gap "
                    f"({gap:.2f}s); resetting trackers",
                    flush=True,
                )

                self.reset_tracking(
                    reason=f"Large frame gap {gap:.2f}s"
                )

            last_seq = seq
            last_received_at = received_at
            last_sample = now

            try:

                boxes = self.tracker.update(
                    frame,
                    received_at,
                )

                for name, bbox in boxes.items():
                    state = self.tracker.targets[name]
                    analyzer = self.analyzers[name]
                    generation_changed = state.generation != self.last_generations[name]
                    if generation_changed:
                        self.last_generations[name] = state.generation
                    som_bbox = self.guards[name].accept(
                        frame, received_at, bbox, generation_changed, analyzer
                    )
                    # 의심 좌표는 None으로 보내어 이동량에 섞지 않음
                    # 추적 시작 전 불필요한 10초 공백 버퍼를 만들지 않음
                    if som_bbox is not None or analyzer.samples:
                        analyzer.add(received_at, som_bbox, name)

                self.update_results()

                self.visualizer.update(
                    frame,
                    boxes,
                    self.analyzers,
                    self.tracker.targets,
                )

            except Exception as exc:

                print(
                    f"[ACTIVITY] Processing error: {exc}",
                    flush=True,
                )

                self.reset_tracking(
                    reason=f"Processing error: {exc}"
                )

                self.stop_event.wait(0.2)

        self.visualizer.clear()

        print(
            "[ACTIVITY] stopped",
            flush=True,
        )
