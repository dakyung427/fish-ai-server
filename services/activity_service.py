
"""SOM 10-second window analysis using shared RTSPCamera."""

import math
import time
import threading

import numpy as np


# ==========================================
# SOM 분석 설정
# ==========================================
WINDOW_SECONDS = 10.0
JITTER_DEADBAND_BODY = 0.015
SMOOTH_RADIUS = 3


# ==========================================
# SOM 분석기
# ==========================================
class SOMAnalyzer:
    """
    10초 단위 SOM 분석

    이상: WARNING +1
    정상: WARNING -1
    WARNING 범위: 0~10
    """

    def __init__(self, model_path):

        m = np.load(model_path, allow_pickle=False)

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

        self.warning = 0
        self.final_abnormal = False

        self.last_score = None
        self.last_features = None
        self.last_result = "WAITING"

    # ======================================
    # 분석 창 초기화
    # ======================================
    def reset(self):

        # 추적이 끊겨도 WARNING은 유지
        self.samples.clear()
        self.start = None

        self.final_abnormal = self.warning > 0
        self.last_result = "PAUSED"

        self.last_score = None
        self.last_features = None

    # ======================================
    # 좌표 데이터 추가 + 10초 분석
    # ======================================
    def add(self, timestamp, bbox, target_name="UNKNOWN"):

        if self.start is None:
            self.start = timestamp

        if bbox is None:

            self.samples.append((
                timestamp,
                float("nan"),
                float("nan"),
                float("nan"),
                0
            ))

        else:

            bx, by, bw, bh = bbox

            x = bx + bw / 2
            y = by + bh / 2

            size = math.hypot(bw, bh)

            self.samples.append((
                timestamp,
                float(x),
                float(y),
                size,
                1
            ))

        if timestamp - self.start < WINDOW_SECONDS:
            return

        row = self.features_for_window(self.samples)

        self.samples = []
        self.start = None

        # 추적 데이터가 부족하면 WARNING 유지
        if row is None:

            self.last_result = "INSUFFICIENT TRACKING"

            print(
                f"[SOM][{target_name.upper()}] "
                f"insufficient tracking; "
                f"WARNING={self.warning} (held)",
                flush=True
            )

            return

        # ==================================
        # SOM 점수 계산
        # ==================================
        raw = np.array(
            [row[name] for name in self.features],
            dtype=float
        )

        z = np.clip(
            (raw - self.mean) / self.scale,
            -8,
            8
        )

        score = float(
            np.sqrt(
                np.min(
                    np.sum(
                        (self.weights - z) ** 2,
                        axis=2
                    )
                )
            )
        )

        abnormal = score > self.threshold

        # ==================================
        # WARNING 누적
        # ==================================
        if abnormal:
            self.warning = min(10, self.warning + 1)
        else:
            self.warning = max(0, self.warning - 1)

        self.final_abnormal = self.warning > 0

        self.last_score = score
        self.last_features = row

        self.last_result = (
            "WARNING"
            if self.final_abnormal
            else "NORMAL"
        )

        print(
            f"[SOM][{target_name.upper()}] "
            f"score={score:.4f} "
            f"threshold={self.threshold:.4f} "
            f"still={row['still_ratio']:.2%} "
            f"distance={row['distance_body']:.3f} "
            f"window={'ABNORMAL' if abnormal else 'NORMAL'} "
            f"WARNING={self.warning} "
            f"status={self.last_result}",
            flush=True
        )

    # ======================================
    # 10초 특징 추출
    # ======================================
    @staticmethod
    def features_for_window(samples):

        if not samples:
            return None

        valid = [
            s for s in samples
            if s[4]
        ]

        coverage = len(valid) / len(samples)

        if coverage < 0.8 or len(valid) < 5:
            return None

        times = np.array(
            [s[0] for s in valid]
        )

        xy = np.array(
            [[s[1], s[2]] for s in valid],
            dtype=float
        )

        body = np.array(
            [s[3] for s in valid],
            dtype=float
        )

        # ==================================
        # 좌표 흔들림 완화
        # ==================================
        smooth = xy.copy()

        for i in range(len(xy)):

            start = max(
                0,
                i - SMOOTH_RADIUS
            )

            end = min(
                len(xy),
                i + SMOOTH_RADIUS + 1
            )

            smooth[i] = np.median(
                xy[start:end],
                axis=0
            )

        intervals = np.diff(times)

        positive = intervals[
            intervals > 0
        ]

        if not len(positive):
            return None

        expected_dt = float(
            np.median(positive)
        )

        distance = 0.0
        still_dt = 0.0
        observed_dt = 0.0

        # ==================================
        # 이동거리 / 정지시간 계산
        # ==================================
        for i in range(1, len(valid)):

            dt = times[i] - times[i - 1]

            if (
                dt <= 0
                or dt > max(
                    0.7,
                    expected_dt * 2.5
                )
            ):
                continue

            unit = max(
                1.0,
                (body[i] + body[i - 1]) / 2
            )

            d = math.dist(
                smooth[i],
                smooth[i - 1]
            ) / unit

            # 미세한 CSRT 흔들림 무시
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
            smooth[0]
        ) / max(
            1.0,
            float(np.median(body))
        )

        if net < JITTER_DEADBAND_BODY:
            net = 0.0

        return {
            "distance_body": distance,
            "speed_body_s": distance / observed_dt,
            "still_ratio": still_dt / observed_dt,
            "net_displacement_body": net
        }


# ==========================================
# 실시간 활동성 분석 서비스
# ==========================================
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

    # ======================================
    # 서비스 종료 요청
    # ======================================
    def stop(self):
        self.stop_event.set()

    # ======================================
    # WebSocket 전송용 결과
    # ======================================
    def snapshot(self):

        with self.results_lock:

            return {
                name: data.copy()
                for name, data in self.last_results.items()
            }

    # ======================================
    # 추적 초기화 (WARNING 유지)
    # ======================================
    def reset_tracking(self):

        self.tracker.reset()

        for analyzer in self.analyzers.values():
            analyzer.reset()

        self.last_generations = {
            name: state.generation
            for name, state in self.tracker.targets.items()
        }

    # ======================================
    # 실시간 분석 실행
    # ======================================
    def run_sync(self):

        print(
            "[ACTIVITY] CSRT + SOM started",
            flush=True
        )

        last_seq = -1
        last_sample = 0.0
        last_received_at = None
        stream_lost = False

        while not self.stop_event.is_set():

            now = time.monotonic()

            # 최대 약 10FPS
            if now - last_sample < 0.10:
                self.stop_event.wait(0.01)
                continue

            # ==================================
            # RTSP 최신 프레임 가져오기
            # ==================================
            frame, seq, received_at, connected = (
                self.camera.read_latest()
            )

            # RTSP 연결 끊김 / 오래된 프레임
            stale = (
                frame is None
                or not connected
                or now - received_at > 2.0
            )

            if stale:

                if not stream_lost:

                    print(
                        "[RTSP] Stream unavailable; "
                        "resetting trackers",
                        flush=True
                    )

                    self.reset_tracking()
                    stream_lost = True

                    # 이전 분석 결과를 PAUSED로 갱신
                    self.update_results()

                self.stop_event.wait(0.05)
                continue

            # 같은 프레임 반복 처리 방지
            if seq == last_seq:
                self.stop_event.wait(0.01)
                continue

            # ==================================
            # RTSP 연결 복구
            # ==================================
            if stream_lost:

                print(
                    "[RTSP] Stream recovered",
                    flush=True
                )

                stream_lost = False
                last_received_at = None

            # 프레임 간격 2초 이상
            if (
                last_received_at is not None
                and received_at - last_received_at > 2.0
            ):

                print(
                    "[RTSP] Large frame gap; "
                    "resetting trackers",
                    flush=True
                )

                self.reset_tracking()
                self.update_results()

            last_seq = seq
            last_received_at = received_at
            last_sample = now

            try:

                # ==================================
                # YOLO + CSRT + HSV 추적
                # ==================================
                boxes = self.tracker.update(
                    frame,
                    received_at
                )

                # ==================================
                # 어종별 SOM 분석
                # ==================================
                for name, bbox in boxes.items():

                    state = self.tracker.targets[name]
                    analyzer = self.analyzers[name]

                    # 추적기가 재초기화되었으면
                    # 분석 창만 초기화
                    if (
                        state.generation
                        != self.last_generations[name]
                    ):

                        analyzer.reset()

                        self.last_generations[name] = (
                            state.generation
                        )

                    # 실제 영상 프레임 시각 사용
                    analyzer.add(
                        received_at,
                        bbox,
                        name
                    )

                self.update_results()

            except Exception as exc:

                print(
                    f"[ACTIVITY] Processing error: {exc}",
                    flush=True
                )

                self.reset_tracking()
                self.update_results()

                self.stop_event.wait(0.2)

        print(
            "[ACTIVITY] stopped",
            flush=True
        )

    # ======================================
    # 분석 결과 업데이트
    # ======================================
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
                )
            }

        with self.results_lock:
            self.last_results = results
