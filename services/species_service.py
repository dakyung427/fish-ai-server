
import time
from collections import defaultdict, Counter

from config import SPECIES_NAME_KO


# ==========================================
# 어종 분석 설정
# ==========================================

ANALYSIS_TIME = 3.0

# 전체 분석 프레임 중 50% 이상 검출되면 인정
MIN_DETECTION_RATIO = 0.50


class SpeciesService:

    def __init__(self, detector, analysis_time=ANALYSIS_TIME):
        self.detector = detector
        self.analysis_time = analysis_time

    # ==========================================
    # 어종 분석
    # ==========================================
    def analyze(self, camera):

        track_history = defaultdict(list)
        total_frames = 0

        start_time = time.monotonic()

        print(
            f"[SPECIES] 어종 분석 시작 "
            f"({self.analysis_time}초)",
            flush=True
        )

        # ==========================================
        # YOLO + ByteTrack 결과 수집
        # ==========================================
        while time.monotonic() - start_time < self.analysis_time:

            frame = camera.read()

            if frame is None:
                time.sleep(0.01)
                continue

            results = self.detector.detect(frame)

            if not results:
                continue

            total_frames += 1

            boxes = results[0].boxes

            if boxes is None or boxes.id is None:
                continue

            # 동일 분석 프레임에서 ID 중복 집계 방지
            detected_ids = set()

            for i in range(len(boxes)):

                track_id = int(boxes.id[i].item())
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())

                if confidence < self.detector.conf:
                    continue

                if track_id in detected_ids:
                    continue

                detected_ids.add(track_id)

                class_name_en = self.detector.model.names[
                    class_id
                ]

                class_name_ko = SPECIES_NAME_KO.get(
                    class_name_en,
                    class_name_en
                )

                track_history[track_id].append({
                    "className": class_name_ko,
                    "confidence": confidence
                })

        # ==========================================
        # 검출 안정성 검증
        # ==========================================
        final_detections = []

        print(
            f"[SPECIES] 총 분석 프레임: {total_frames}",
            flush=True
        )

        for track_id, records in track_history.items():

            detection_count = len(records)

            detection_ratio = (
                detection_count / total_frames
                if total_frames > 0
                else 0.0
            )

            # ======================================
            # 검출 비율 검증
            # 최소 검출 횟수 제한 없음
            # ======================================
            if detection_ratio < MIN_DETECTION_RATIO:

                print(
                    f"[SPECIES] ID={track_id} 제외 "
                    f"(검출={detection_count}/{total_frames}, "
                    f"비율={detection_ratio:.1%})",
                    flush=True
                )
                continue

            # ======================================
            # Track ID별 최빈 어종 결정
            # ======================================
            class_counter = Counter(
                record["className"]
                for record in records
            )

            final_class = class_counter.most_common(1)[0][0]

            confidences = [
                record["confidence"]
                for record in records
                if record["className"] == final_class
            ]

            avg_confidence = (
                sum(confidences) / len(confidences)
            )

            final_detections.append({
                "trackId": track_id,
                "className": final_class,
                "confidence": round(avg_confidence, 4)
            })

            print(
                f"[SPECIES] ID={track_id} 인정 "
                f"어종={final_class} "
                f"검출={detection_count}/{total_frames} "
                f"비율={detection_ratio:.1%} "
                f"confidence={avg_confidence:.4f}",
                flush=True
            )

        # ==========================================
        # 어종별 개체 수 집계
        # ==========================================
        species_count = Counter(
            detection["className"]
            for detection in final_detections
        )

        result = {
            "totalFishCount": len(final_detections),
            "speciesCount": dict(species_count),
            "detections": final_detections
        }

        print("[SPECIES] 어종 분석 완료", flush=True)
        print(result, flush=True)

        return result
