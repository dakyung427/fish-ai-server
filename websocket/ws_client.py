
import asyncio
import json

import websockets

from config import NODE_WS_URL


# 실제 어항 기기번호
DEVICE_ID = "SS501"

# SOM 상태 확인 간격
ACTIVITY_CHECK_INTERVAL = 0.5


class WebSocketClient:

    def __init__(
        self,
        species_service,
        camera,
        activity_service=None,
        url=NODE_WS_URL
    ):
        self.url = url
        self.websocket = None
        self.running = False

        self.species_service = species_service
        self.activity_service = activity_service
        self.camera = camera

        self.send_lock = asyncio.Lock()

        self.activity_task = None
        self.species_tasks = set()
        self.species_busy = False

        # 마지막으로 전송한 SOM 상태
        self.last_activity_signature = None

    # ==========================================
    # WebSocket 서버 연결
    # ==========================================
    async def connect(self):

        print(f"WebSocket 연결 시도: {self.url}")

        self.websocket = await websockets.connect(
            self.url
        )

        print("WebSocket 연결 성공")

        # AI 서버 등록
        await self.send_json({
            "type": "Ai"
        })

        print("AI 서버 등록 메시지 전송 완료")

        # 재연결 시 현재 상태 재전송 허용
        self.last_activity_signature = None

    # ==========================================
    # JSON 메시지 전송
    # ==========================================
    async def send_json(self, data):

        message = json.dumps(
            data,
            ensure_ascii=False
        )

        async with self.send_lock:

            if self.websocket is None:
                print("WebSocket 연결 안 됨")
                return False

            await self.websocket.send(message)

        print("[WS] 전송:", message)

        return True

    # ==========================================
    # 어종 분석 작업
    # ==========================================
    async def run_species_analysis(self, device_id):

        try:

            print(
                f"[SPECIES] 분석 시작: {device_id}"
            )

            # 이벤트 루프 차단 방지
            result = await asyncio.to_thread(
                self.species_service.analyze,
                self.camera
            )

            await self.send_json({
                "type": "species_result",
                "device_id": device_id,
                "data": result
            })

            print("[SPECIES] 분석 결과 전송 완료")

        except Exception as exc:

            print(
                f"[SPECIES] 분석 오류: {exc}",
                flush=True
            )

        finally:
            self.species_busy = False

    # ==========================================
    # SOM 활동성 상태 집계
    # ==========================================
    def get_activity_result(self):

        if self.activity_service is None:
            return None

        snapshot = self.activity_service.snapshot()

        if not snapshot:
            return None

        healthy = 0
        abnormal = 0
        detail = []

        # 실제 확인된 어종별 상태
        signature = []

        for name, info in sorted(snapshot.items()):

            state = info.get("state")
            status = info.get("status")

            # CSRT가 정상 추적 중이어야 함
            if state != "TRACKING":
                continue

            # SOM 분석이 완료된 상태만 집계
            if status not in ("NORMAL", "WARNING"):
                continue

            is_abnormal = bool(
                info.get("abnormal", False)
            )

            if is_abnormal:
                abnormal += 1
            else:
                healthy += 1

            signature.append((
                name,
                is_abnormal
            ))

            detail.append({
                "className": name,
                "status": (
                    "abnormal"
                    if is_abnormal
                    else "healthy"
                )
            })

        # 확인된 상태가 없다면 전송하지 않음
        if not signature:
            return None

        result = {
            "healthy": healthy,
            "abnormal": abnormal,
            "detail": detail
        }

        return result, tuple(signature)

    # ==========================================
    # SOM 상태 변경 감지 및 전송
    # ==========================================
    async def activity_send_loop(self):

        print("[ACTIVITY] 상태 감시 시작")

        while self.running:

            try:

                activity = self.get_activity_result()

                if activity is not None:

                    result, signature = activity

                    # 최초 전송 또는 상태 변경
                    if signature != self.last_activity_signature:

                        sent = await self.send_json({
                            "type": "activity_result",
                            "device_id": DEVICE_ID,
                            "data": result
                        })

                        if sent:
                            self.last_activity_signature = signature

                            print(
                                "[ACTIVITY] 상태 변경 전송:",
                                result,
                                flush=True
                            )

            except asyncio.CancelledError:
                raise

            except Exception as exc:

                print(
                    f"[ACTIVITY] 전송 오류: {exc}",
                    flush=True
                )
                raise

            await asyncio.sleep(
                ACTIVITY_CHECK_INTERVAL
            )

    # ==========================================
    # 메시지 수신
    # ==========================================
    async def receive_loop(self):

        async for message in self.websocket:

            try:

                data = json.loads(message)

                print("[WS] 수신:", data)

                await self.handle_message(data)

            except json.JSONDecodeError:

                print(
                    "잘못된 JSON 메시지:",
                    message
                )

    # ==========================================
    # Node.js 메시지 처리
    # ==========================================
    async def handle_message(self, data):

        message_type = data.get("type")

        # ======================================
        # 어종·개체수 분석 요청
        # ======================================
        if message_type == "species":

            device_id = data.get("device_id")

            if not device_id:

                print(
                    "species 요청에 device_id가 없습니다."
                )
                return

            if self.species_busy:

                print(
                    "[SPECIES] 이미 분석 중입니다."
                )
                return

            self.species_busy = True

            task = asyncio.create_task(
                self.run_species_analysis(device_id)
            )

            self.species_tasks.add(task)

            task.add_done_callback(
                self.species_tasks.discard
            )

        else:

            print(
                "알 수 없는 메시지 type:",
                message_type
            )

    # ==========================================
    # WebSocket 실행 / 자동 재연결
    # ==========================================
    async def run(self):

        self.running = True

        while self.running:

            try:

                await self.connect()

                # SOM 상태 자동 전송
                if self.activity_service is not None:

                    self.activity_task = asyncio.create_task(
                        self.activity_send_loop()
                    )

                await self.receive_loop()

            except asyncio.CancelledError:
                raise

            except Exception as exc:

                print(
                    f"WebSocket 연결 오류: {exc}",
                    flush=True
                )

            finally:

                # 이전 연결의 SOM 전송 작업 종료
                if self.activity_task is not None:

                    self.activity_task.cancel()

                    try:
                        await self.activity_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        print(
                            f"[ACTIVITY] 작업 종료: {exc}"
                        )

                    self.activity_task = None

                if self.websocket is not None:

                    await self.websocket.close()
                    self.websocket = None

            if self.running:

                print("3초 후 WebSocket 재연결...")

                await asyncio.sleep(3)

    # ==========================================
    # WebSocket 종료
    # ==========================================
    async def stop(self):

        self.running = False

        if self.activity_task is not None:
            self.activity_task.cancel()

        for task in list(self.species_tasks):
            task.cancel()

        if self.websocket is not None:
            await self.websocket.close()

        print("WebSocket 종료")
