"""10-second SOM activity analysis from a shared, sequence-numbered RTSP camera."""
import math
import threading
import time

import numpy as np

from vision.activity_visualizer import ActivityVisualizer

WINDOW_SECONDS = 10.0
JITTER_DEADBAND_BODY = 0.015
SMOOTH_RADIUS = 3


class SOMAnalyzer:
    def __init__(self, model_path):
        with np.load(model_path, allow_pickle=False) as m:
            self.weights = m['weights']
            self.mean = m['mean']
            self.scale = m['scale']
            self.threshold = float(m['threshold'])
            self.features = [str(v) for v in m['features']]
        expected = ['distance_body', 'speed_body_s', 'still_ratio', 'net_displacement_body']
        if self.features != expected:
            raise ValueError(f'Unexpected SOM features: {self.features}')
        self.samples = []
        self.start = None
        self.warning = 0
        self.final_abnormal = False
        self.last_score = None
        self.last_features = None
        self.last_result = 'WAITING'

    def reset(self):
        self.samples.clear()
        self.start = None
        self.final_abnormal = self.warning > 0
        self.last_result = 'PAUSED'
        self.last_score = None
        self.last_features = None

    def add(self, timestamp, bbox, target_name='UNKNOWN'):
        if self.start is None:
            self.start = timestamp
        if bbox is None:
            self.samples.append((timestamp, float('nan'), float('nan'), float('nan'), 0))
        else:
            bx, by, bw, bh = bbox
            self.samples.append((timestamp, bx + bw / 2, by + bh / 2,
                                 math.hypot(bw, bh), 1))
        if timestamp - self.start < WINDOW_SECONDS:
            return
        row = self.features_for_window(self.samples)
        self.samples = []
        self.start = None
        if row is None:
            self.last_result = 'INSUFFICIENT TRACKING'
            print(f'[SOM][{target_name.upper()}] insufficient tracking; '
                  f'WARNING={self.warning} (held)', flush=True)
            return
        raw = np.array([row[name] for name in self.features], dtype=float)
        z = np.clip((raw - self.mean) / self.scale, -8, 8)
        score = float(np.sqrt(np.min(np.sum((self.weights - z) ** 2, axis=2))))
        abnormal = score > self.threshold
        self.warning = min(10, self.warning + 1) if abnormal else max(0, self.warning - 1)
        self.final_abnormal = self.warning > 0
        self.last_score = score
        self.last_features = row
        self.last_result = 'WARNING' if self.final_abnormal else 'NORMAL'
        print(f'[SOM][{target_name.upper()}] score={score:.4f} threshold={self.threshold:.4f} '
              f'still={row["still_ratio"]:.2%} distance={row["distance_body"]:.3f} '
              f'window={"ABNORMAL" if abnormal else "NORMAL"} '
              f'WARNING={self.warning} status={self.last_result}', flush=True)

    @staticmethod
    def features_for_window(samples):
        if not samples:
            return None
        valid = [s for s in samples if s[4]]
        coverage = len(valid) / len(samples)
        if coverage < 0.8 or len(valid) < 5:
            return None
        times = np.array([s[0] for s in valid])
        xy = np.array([[s[1], s[2]] for s in valid], dtype=float)
        body = np.array([s[3] for s in valid], dtype=float)
        smooth = xy.copy()
        for i in range(len(xy)):
            smooth[i] = np.median(xy[max(0, i - SMOOTH_RADIUS):
                                     min(len(xy), i + SMOOTH_RADIUS + 1)], axis=0)
        intervals = np.diff(times)
        positive = intervals[intervals > 0]
        if not len(positive):
            return None
        expected_dt = float(np.median(positive))
        distance = still_dt = observed_dt = 0.0
        for i in range(1, len(valid)):
            dt = times[i] - times[i - 1]
            if dt <= 0 or dt > max(0.7, expected_dt * 2.5):
                continue
            unit = max(1., (body[i] + body[i - 1]) / 2)
            d = math.dist(smooth[i], smooth[i - 1]) / unit
            if d < JITTER_DEADBAND_BODY:
                d = 0.0
            distance += d
            observed_dt += dt
            if d / dt < 0.035:
                still_dt += dt
        if observed_dt < 5.0:
            return None
        net = math.dist(smooth[-1], smooth[0]) / max(1., float(np.median(body)))
        if net < JITTER_DEADBAND_BODY:
            net = 0.0
        return dict(distance_body=distance, speed_body_s=distance / observed_dt,
                    still_ratio=still_dt / observed_dt, net_displacement_body=net)


class ActivityService:
    def __init__(self, tracker, camera, model_path):
        self.tracker = tracker
        self.camera = camera
        self.analyzers = {name: SOMAnalyzer(model_path) for name in tracker.targets}
        self.stop_event = threading.Event()
        self.results_lock = threading.Lock()
        self.last_results = {}
        self.last_generations = {
            name: state.generation for name, state in tracker.targets.items()
        }
        self.visualizer = ActivityVisualizer()

    def stop(self):
        self.stop_event.set()

    def snapshot(self):
        with self.results_lock:
            return {name: {**data, 'features': data['features'].copy()
                           if data['features'] is not None else None}
                    for name, data in self.last_results.items()}

    def preview_snapshot(self):
        return self.visualizer.get_frame()

    def update_results(self):
        results = {}
        for name, analyzer in self.analyzers.items():
            results[name] = {
                'state': self.tracker.targets[name].state,
                'status': analyzer.last_result,
                'warning': analyzer.warning,
                'abnormal': analyzer.final_abnormal,
                'score': analyzer.last_score,
                'features': analyzer.last_features.copy()
                if analyzer.last_features is not None else None,
            }
        with self.results_lock:
            self.last_results = results

    def reset_tracking(self):
        self.tracker.reset()
        for analyzer in self.analyzers.values():
            analyzer.reset()
        self.last_generations = {
            name: state.generation for name, state in self.tracker.targets.items()
        }
        self.visualizer.clear()
        self.update_results()

    def run_sync(self):
        print('[ACTIVITY] CSRT + SOM started', flush=True)
        last_seq = -1
        last_sample = 0.0
        last_received_at = None
        stream_lost = False
        while not self.stop_event.is_set():
            now = time.monotonic()
            if now - last_sample < 0.10:
                self.stop_event.wait(0.01)
                continue
            frame, seq, received_at, connected = self.camera.read_latest()
            stale = frame is None or not connected or now - received_at > 2.0
            if stale:
                if not stream_lost:
                    print('[RTSP] Stream unavailable; resetting trackers', flush=True)
                    self.reset_tracking()
                    stream_lost = True
                self.stop_event.wait(0.05)
                continue
            if seq == last_seq:
                self.stop_event.wait(0.01)
                continue
            if stream_lost:
                print('[RTSP] Stream recovered', flush=True)
                stream_lost = False
                last_received_at = None
            if (last_received_at is not None
                    and received_at - last_received_at > 2.0):
                print('[RTSP] Large frame gap; resetting trackers', flush=True)
                self.reset_tracking()
            last_seq = seq
            last_received_at = received_at
            last_sample = now
            try:
                boxes = self.tracker.update(frame, received_at)
                for name, bbox in boxes.items():
                    state = self.tracker.targets[name]
                    analyzer = self.analyzers[name]
                    if state.generation != self.last_generations[name]:
                        analyzer.reset()
                        self.last_generations[name] = state.generation
                    analyzer.add(received_at, bbox, name)
                self.update_results()
                self.visualizer.update(frame, boxes, self.analyzers, self.tracker.targets)
            except Exception as exc:
                print(f'[ACTIVITY] Processing error: {exc}', flush=True)
                self.reset_tracking()
                self.stop_event.wait(0.2)
        self.visualizer.clear()
        print('[ACTIVITY] stopped', flush=True)
