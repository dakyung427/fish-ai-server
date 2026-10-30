
import asyncio
import json
import time

import websockets

from config import NODE_WS_URL
from services.activity_result_service import ActivityResultService


# ==========================================
# 설정
# ==========================================

# 실제 어항 기기번호
DEVICE_ID = "SS501"

# SOM 상태 확인 간격 (초)
ACTIVITY_CHECK_INTERVAL = 0.5

# SOM 결과 주기적 재전송 간격 (초)
ACTIVITY_RESEND_INTERVAL = 10.0


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

        # ==========================================
        # SOM 활동성 결과 집계 서비스
        # ==========================================
        self.activity_result_service = (
            ActivityResultService(activity_service)
            if activity_service is not None
            else None
        )

        self.send_lock = asyncio.Lock()

        self.activity_task = None
        self.species_tasks = set()
        self.species_busy = False

        # ==========================================
        # 마지막 SOM 전송 상태
        # ==========================================
        self.last_activity_signature = None
        self.last_activity_sent_at = 0.0

    # ==========================================
    # WebSocket 서버 연결
    # ==========================================
    async def connect(self):

        print(
            f"WebSocket 연결 시도: {self.url}",
            flush=True
        )

        self.websocket = await websockets.connect(
            self.url
        )

        print("WebSocket 연결 성공", flush=True)

        # AI 서버 등록
        await self.send_json({
            "type": "Ai"
        })

        print(
            "AI 서버 등록 메시지 전송 완료",
            flush=True
        )

        # ======================================
        # 재연결 시 현재 SOM 상태 재전송 허용
        # ======================================
        self.last_activity_signature = None
        self.last_activity_sent_at = 0.0

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
                print(
                    "WebSocket 연결 안 됨",
                    flush=True
                )
                return False

            await self.websocket.send(message)

        print(
            "[WS] 전송:",
            message,
            flush=True
        )

        return True

    # ==========================================
    # 어종 분석 작업
    # ==========================================
    async def run_species_analysis(self, device_id):

        try:

            print(
                f"[SPECIES] 분석 시작: {device_id}",
                flush=True
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

            print(
                "[SPECIES] 분석 결과 전송 완료",
                flush=True
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:

            print(
                f"[SPECIES] 분석 오류: {exc}",
                flush=True
            )

        finally:
            self.species_busy = False

    # ==========================================
    # SOM 활동성 결과 집계
    # ==========================================
    def get_activity_result(self):

        if self.activity_result_service is None:
            return None

        # 별도 서비스에서 건강/주의 개체 수 집계
        # 일시적인 추적 재초기화 시 마지막 유효 판정 유지
        return self.activity_result_service.get_result()

    # ==========================================
    # SOM 상태 변경 감지 및 주기적 전송
    # ==========================================
    async def activity_send_loop(self):

        print(
            "[ACTIVITY] 상태 감시 시작",
            flush=True
        )

        while self.running:

            try:

                activity = self.get_activity_result()

                if activity is not None:

                    result, signature = activity

                    now = time.monotonic()

                    # ==================================
                    # 전송 조건
                    # 1. 최초 전송
                    # 2. 건강/주의 상태 변경
                    # 3. 마지막 전송 후 10초 경과
                    # ==================================
                    state_changed = (
                        signature != self.last_activity_signature
                    )

                    resend_due = (
                        now - self.last_activity_sent_at
                        >= ACTIVITY_RESEND_INTERVAL
                    )

                    if state_changed or resend_due:

                        sent = await self.send_json({
                            "type": "activity_result",
                            "device_id": DEVICE_ID,
                            "data": result
                        })

                        if sent:

                            self.last_activity_signature = signature
                            self.last_activity_sent_at = (
                                time.monotonic()
                            )

                            if state_changed:
                                reason = "상태 변경"
                            else:
                                reason = "10초 주기 재전송"

                            print(
                                f"[ACTIVITY] {reason}: "
                                f"건강={result['healthy']} "
                                f"주의={result['abnormal']}",
                                flush=True
                            )

            except asyncio.CancelledError:
                raise

            except Exception as exc:

                print(
                    f"[ACTIVITY] 전송 오류: {exc}",
                    flush=True
                )

                # 오류 발생 시 재연결 처리
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

                print(
                    "[WS] 수신:",
                    data,
                    flush=True
                )

                await self.handle_message(data)

            except json.JSONDecodeError:

                print(
                    "잘못된 JSON 메시지:",
                    message,
                    flush=True
                )

    # ==========================================
    # Node.js 메시지 처리
    # ==========================================
    async def handle_message(self, data):

        message_type = data.get("type")

        # ======================================
        # 어종·개체 수 분석 요청
        # ======================================
        if message_type == "species":

            device_id = data.get("device_id")

            if not device_id:

                print(
                    "species 요청에 device_id가 없습니다.",
                    flush=True
                )
                return

            if self.species_busy:

                print(
                    "[SPECIES] 이미 분석 중입니다.",
                    flush=True
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
                message_type,
                flush=True
            )

    # ==========================================
    # WebSocket 실행 / 자동 재연결
    # ==========================================
    async def run(self):

        self.running = True

        while self.running:

            try:

                await self.connect()

                # ==================================
                # SOM 상태 자동 전송 작업 시작
                # ==================================
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

                # ==================================
                # 기존 SOM 전송 작업 종료
                # ==================================
                if self.activity_task is not None:

                    self.activity_task.cancel()

                    try:
                        await self.activity_task

                    except asyncio.CancelledError:
                        pass

                    except Exception as exc:
                        print(
                            f"[ACTIVITY] 작업 종료: {exc}",
                            flush=True
                        )

                    self.activity_task = None

                # ==================================
                # WebSocket 연결 종료
                # ==================================
                if self.websocket is not None:

                    await self.websocket.close()
                    self.websocket = None

            if self.running:

                print(
                    "3초 후 WebSocket 재연결...",
                    flush=True
                )

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

        print(
            "WebSocket 종료",
            flush=True
        )
