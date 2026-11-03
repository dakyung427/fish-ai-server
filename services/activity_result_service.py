"""Goldfish / Fishmodel SOM 결과를 Node.js 전송 형식으로 집계한다."""
import time

TRACK_LOST_HOLD_SECONDS = 5.0
MAX_RESULT_AGE_SECONDS = 15.0  # 마지막 SOM 분석 이후 최대 유지 시간


class ActivityResultService:
    def __init__(self, activity_service):
        self.activity_service = activity_service
        self.last_valid = {}

    def get_result(self):
        snapshot = self.activity_service.snapshot()
        now = time.monotonic()
        healthy, abnormal = 0, 0
        detail, signature = [], []

        for name, info in sorted(snapshot.items()):
            is_tracking = info.get("state") == "TRACKING"
            status = info.get("status")
            analyzed_at = info.get("last_analyzed_at")
            cached = self.last_valid.get(name)

            # NORMAL/WARNING은 새 SOM 분석이 끝난 경우에만 갱신.
            # PAUSED / WAITING / INSUFFICIENT TRACKING은 새 판정이 아니다.
            if (is_tracking and status in ("NORMAL", "WARNING")
                    and analyzed_at is not None
                    and 0 <= now - analyzed_at <= MAX_RESULT_AGE_SECONDS
                    and (cached is None or analyzed_at > cached["analyzed_at"])):
                cached = {
                    "abnormal": status == "WARNING",
                    "analyzed_at": analyzed_at,
                    "last_tracking_at": now,
                }
                self.last_valid[name] = cached
                print(f"[ACTIVITY RESULT] {name} 새 SOM 판정: {status}", flush=True)

            cached = self.last_valid.get(name)
            if cached is None:
                continue

            age = now - cached["analyzed_at"]
            if age > MAX_RESULT_AGE_SECONDS:
                del self.last_valid[name]
                print(f"[ACTIVITY RESULT] {name} 마지막 SOM 판정 만료 ({age:.1f}s)", flush=True)
                continue

            if is_tracking:
                cached["last_tracking_at"] = now
            elif now - cached["last_tracking_at"] > TRACK_LOST_HOLD_SECONDS:
                del self.last_valid[name]
                print(f"[ACTIVITY RESULT] {name} 추적 중단으로 집계 제외", flush=True)
                continue

            is_abnormal = cached["abnormal"]
            abnormal += int(is_abnormal)
            healthy += int(not is_abnormal)
            signature.append((name, is_abnormal))
            detail.append({
                "className": name,
                "status": "abnormal" if is_abnormal else "healthy",
            })

        if not signature:
            return None
        return {
            "healthy": healthy,
            "abnormal": abnormal,
            "detail": detail,
        }, tuple(signature)
