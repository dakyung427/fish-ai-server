
import time

from collections import defaultdict, Counter

from config import SPECIES_NAME_KO


# ==========================================
# 어종 검출 안정성 검증 설정
# ==========================================

# 동일 Track ID 최소 검출 프레임 수
MIN_DETECTION_FRAMES = 8

# 전체 분석 프레임 대비 최소 검출 비율
MIN_DETECTION_RATIO = 0.50

# 분석 시간을 나눌 구간 수
TEMPORAL_SEGMENTS = 3


class SpeciesService:

    def __init__(self, detector, analysis_time=3.0):
        self.detector = detector
        self.analysis_time = analysis_time

    # ==========================================
    # 어종 분석
    # ==========================================
    def analyze(self, camera):

        # Track ID별 검출 기록
        track_history = defaultdict(list)

        # Track ID별 검출된 시간 구간
        track_segments = defaultdict(set)

        # 전체 분석 프레임 수
        total_frames = 0

        start_time = time.monotonic()

        print("어종 분석 시작...", flush=True)

        # ==========================================
        # 3초 동안 YOLO + ByteTrack 결과 수집
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

            # 현재 분석 시간 구간
            elapsed = time.monotonic() - start_time

            segment_duration = (
                self.analysis_time / TEMPORAL_SEGMENTS
            )

            segment_index = min(
                int(elapsed / segment_duration),
                TEMPORAL_SEGMENTS - 1
            )

            result = results[0]
            boxes = result.boxes

            if boxes is None or boxes.id is None:
                continue

            # 동일 프레임에서 같은 Track ID 중복 방지
            detected_ids = set()

            # ==========================================
            # Track ID별 탐지 결과 누적
            # ==========================================
            for i in range(len(boxes)):

                track_id = int(
                    boxes.id[i].item()
                )

                class_id = int(
                    boxes.cls[i].item()
                )

                confidence = float(
                    boxes.conf[i].item()
                )

                # confidence 기준 미달 제외
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

                # 해당 ID가 등장한 시간 구간 기록
                track_segments[track_id].add(
                    segment_index
                )

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

            if detection_count == 0:
                continue

            detection_ratio = (
                detection_count / total_frames
                if total_frames > 0
                else 0.0
            )

            detected_segments = track_segments[track_id]

            # ======================================
            # 1. 최소 검출 프레임 검증
            # ======================================
            if detection_count < MIN_DETECTION_FRAMES:

                print(
                    f"[SPECIES] ID={track_id} 제외 "
                    f"(검출 횟수 부족: {detection_count}회)",
                    flush=True
                )
                continue

            # ======================================
            # 2. 최소 검출 비율 검증
            # ======================================
            if detection_ratio < MIN_DETECTION_RATIO:

                print(
                    f"[SPECIES] ID={track_id} 제외 "
                    f"(검출 비율 부족: {detection_ratio:.1%})",
                    flush=True
                )
                continue

            # ======================================
            # 3. 시간대별 지속성 검증
            # ======================================
            if len(detected_segments) < TEMPORAL_SEGMENTS:

                print(
                    f"[SPECIES] ID={track_id} 제외 "
                    f"(지속성 부족: "
                    f"{len(detected_segments)}/{TEMPORAL_SEGMENTS}구간)",
                    flush=True
                )
                continue

            # ======================================
            # 가장 많이 검출된 어종 결정
            # ======================================
            class_names = [
                record["className"]
                for record in records
            ]

            class_counter = Counter(class_names)

            final_class = class_counter.most_common(1)[0][0]

            # 최빈 어종의 평균 confidence
            confidences = [
                record["confidence"]
                for record in records
                if record["className"] == final_class
            ]

            avg_confidence = (
                sum(confidences) / len(confidences)
            )

            # ======================================
            # 검증 통과한 개체만 저장
            # ======================================
            final_detections.append({
                "trackId": track_id,
                "className": final_class,
                "confidence": round(
                    avg_confidence,
                    4
                )
            })

            print(
                f"[SPECIES] ID={track_id} 인정 "
                f"어종={final_class} "
                f"검출={detection_count}회 "
                f"비율={detection_ratio:.1%} "
                f"지속성={len(detected_segments)}/{TEMPORAL_SEGMENTS} "
                f"confidence={avg_confidence:.4f}",
                flush=True
            )

        # ==========================================
        # 어종별 개체 수 계산
        # ==========================================
        species_count = Counter(
            detection["className"]
            for detection in final_detections
        )

        # ==========================================
        # 최종 결과
        # ==========================================
        result = {
            "totalFishCount": len(final_detections),
            "speciesCount": dict(species_count),
            "detections": final_detections
        }

        print("어종 분석 완료", flush=True)
        print(result, flush=True)

        return result
