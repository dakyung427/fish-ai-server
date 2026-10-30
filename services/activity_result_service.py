
import time


# ==========================================
# 활동성 결과 유지 설정
# ==========================================

# 추적이 끊겼을 때 마지막 판정 유지시간
TRACK_LOST_HOLD_SECONDS = 5.0

# SOM 분석 결과 최대 유효시간
# 반복 재탐지로 인한 일시적 PAUSED 상태를 고려
MAX_RESULT_AGE_SECONDS = 120.0


class ActivityResultService:

    def __init__(self, activity_service):

        self.activity_service = activity_service

        # 물고기별 마지막 유효 SOM 판정 저장
        self.last_valid = {}

    # ==========================================
    # 건강 / 주의 물고기 집계
    # ==========================================
    def get_result(self):

        snapshot = self.activity_service.snapshot()
        now = time.monotonic()

        healthy = 0
        abnormal = 0
        detail = []
        signature = []

        for name, info in sorted(snapshot.items()):

            state = info.get("state")
            status = info.get("status")
            analyzed_at = info.get("last_analyzed_at")

            is_tracking = state == "TRACKING"

            cached = self.last_valid.get(name)

            # ==================================
            # 1. 새로운 SOM 판정 저장
            # ==================================
            if (
                is_tracking
                and status in ("NORMAL", "WARNING")
                and analyzed_at is not None
                and 0 <= now - analyzed_at <= MAX_RESULT_AGE_SECONDS
            ):

                # 실제 새 분석 결과가 나왔을 때만 갱신
                if (
                    cached is None
                    or analyzed_at > cached["analyzed_at"]
                ):

                    cached = {
                        "abnormal": bool(
                            info.get("abnormal", False)
                        ),
                        "analyzed_at": analyzed_at,
                        "last_tracking_at": now,
                    }

                    self.last_valid[name] = cached

                    print(
                        f"[ACTIVITY RESULT] {name} "
                        f"새 SOM 판정: {status}",
                        flush=True
                    )

            # ==================================
            # 2. 마지막 확정 판정 확인
            # ==================================
            cached = self.last_valid.get(name)

            if cached is None:
                continue

            # ==================================
            # 3. 추적 중이면 추적 시각 갱신
            # ==================================
            if is_tracking:
                cached["last_tracking_at"] = now

            # ==================================
            # 4. SOM 판정 유효시간 확인
            # ==================================
            result_age = now - cached["analyzed_at"]

            if result_age > MAX_RESULT_AGE_SECONDS:

                print(
                    f"[ACTIVITY RESULT] {name} "
                    f"마지막 SOM 판정 만료 "
                    f"({result_age:.1f}초)",
                    flush=True
                )

                del self.last_valid[name]
                continue

            # ==================================
            # 5. 추적 중단 유예시간 확인
            # ==================================
            tracking_gap = (
                now - cached["last_tracking_at"]
            )

            if (
                not is_tracking
                and tracking_gap > TRACK_LOST_HOLD_SECONDS
            ):

                print(
                    f"[ACTIVITY RESULT] {name} "
                    f"추적 중단 {tracking_gap:.1f}초 "
                    f"-> 집계 제외",
                    flush=True
                )

                del self.last_valid[name]
                continue

            # ==================================
            # 6. PAUSED일 때 마지막 판정 유지
            # ==================================
            if (
                status not in ("NORMAL", "WARNING")
                and is_tracking
            ):
                # 기존 SOM 판정을 사용
                # 매번 로그 출력하지 않음
                pass

            # ==================================
            # 7. 최종 건강 / 주의 집계
            # ==================================
            final_abnormal = cached["abnormal"]

            if final_abnormal:
                abnormal += 1
            else:
                healthy += 1

            signature.append((
                name,
                final_abnormal
            ))

            detail.append({
                "className": name,
                "status": (
                    "abnormal"
                    if final_abnormal
                    else "healthy"
                )
            })

        # ==========================================
        # 8. 유효한 판정이 없는 경우
        # ==========================================
        if not signature:
            return None

        # ==========================================
        # 9. 최종 앱 전송 결과
        # ==========================================
        result = {
            "healthy": healthy,
            "abnormal": abnormal,
            "detail": detail
        }

        return result, tuple(signature)
