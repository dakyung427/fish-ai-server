import time

from collections import defaultdict, Counter

from config import SPECIES_NAME_KO


class SpeciesService:
    def __init__(self, detector, analysis_time=3.0):
        self.detector = detector
        self.analysis_time = analysis_time

    # ==========================================
    # 어종 분석
    # ==========================================
    def analyze(self, camera):

        # trackId별 탐지 결과 저장
        track_history = defaultdict(list)

        start_time = time.time()

        print("어종 분석 시작...")


        # ==========================================
        # 일정 시간 동안 YOLO + ByteTrack 결과 수집
        # ==========================================
        while time.time() - start_time < self.analysis_time:

            # 최신 프레임 가져오기
            frame = camera.read()

            if frame is None:
                time.sleep(0.01)
                continue


            # YOLO + ByteTrack 실행
            results = self.detector.detect(frame)

            result = results[0]
            boxes = result.boxes


            # 탐지 결과가 없으면 다음 프레임
            if boxes is None:
                continue


            # ByteTrack ID가 아직 없는 경우
            if boxes.id is None:
                continue


            # ==========================================
            # ID별 탐지 결과 누적
            # ==========================================
            for i in range(len(boxes)):

                # ByteTrack ID
                track_id = int(
                    boxes.id[i].item()
                )


                # YOLO 클래스 번호
                class_id = int(
                    boxes.cls[i].item()
                )


                # confidence
                confidence = float(
                    boxes.conf[i].item()
                )


                # confidence 기준보다 낮으면 제외
                if confidence < self.detector.conf:
                    continue


                # 모델의 영어 클래스명
                class_name_en = self.detector.model.names[
                    class_id
                ]


                # 영어 클래스명 → 한글 클래스명
                class_name_ko = SPECIES_NAME_KO.get(
                    class_name_en,
                    class_name_en
                )


                # 해당 trackId의 기록에 추가
                track_history[track_id].append(
                    {
                        "className": class_name_ko,
                        "confidence": confidence
                    }
                )


        # ==========================================
        # ID별 최종 어종 결정
        # ==========================================

        final_detections = []


        for track_id, records in track_history.items():

            if not records:
                continue


            # 해당 ID에서 등장한 어종 이름들
            class_names = [
                record["className"]
                for record in records
            ]


            # 가장 많이 등장한 어종 선택
            class_counter = Counter(class_names)

            final_class = class_counter.most_common(1)[0][0]


            # 최종 선택된 어종의 confidence만 추출
            confidences = [
                record["confidence"]
                for record in records
                if record["className"] == final_class
            ]


            # 평균 confidence 계산
            avg_confidence = (
                sum(confidences) / len(confidences)
            )


            # 최종 개체 결과 저장
            final_detections.append(
                {
                    "trackId": track_id,
                    "className": final_class,
                    "confidence": round(
                        avg_confidence,
                        4
                    )
                }
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


        print("어종 분석 완료")
        print(result)


        return result