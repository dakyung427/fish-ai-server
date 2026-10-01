import asyncio
import json

import websockets

from config import NODE_WS_URL


class WebSocketClient:
    def __init__(
        self,
        species_service,
        camera,
        url=NODE_WS_URL
    ):
        self.url = url
        self.websocket = None
        self.running = False

        self.species_service = species_service
        self.camera = camera

    # ==========================================
    # WebSocket 서버 연결
    # ==========================================
    async def connect(self):
        print(f"WebSocket 연결 시도: {self.url}")

        self.websocket = await websockets.connect(
            self.url
        )

        print("WebSocket 연결 성공")

        # Node.js에 AI 서버로 등록
        await self.send_json(
            {
                "type": "Ai"
            }
        )

        print("AI 서버 등록 메시지 전송 완료")

    # ==========================================
    # JSON 메시지 전송
    # ==========================================
    async def send_json(self, data):

        if self.websocket is None:
            print("WebSocket 연결 안 됨")
            return

        message = json.dumps(
            data,
            ensure_ascii=False
        )

        await self.websocket.send(message)

        print("WebSocket 전송:")
        print(message)

    # ==========================================
    # 메시지 수신
    # ==========================================
    async def receive_loop(self):

        async for message in self.websocket:

            try:
                data = json.loads(message)

                print("WebSocket 수신:")
                print(data)

                await self.handle_message(data)

            except json.JSONDecodeError:
                print(
                    "잘못된 JSON 메시지:",
                    message
                )

    # ==========================================
    # 받은 메시지 처리
    # ==========================================
    async def handle_message(self, data):

        message_type = data.get("type")

        # ======================================
        # 어종 분석 요청
        # Node → AI
        # {
        #   "type": "species",
        #   "device_id": "..."
        # }
        # ======================================
        if message_type == "species":

            device_id = data.get("device_id")

            if not device_id:
                print(
                    "species 요청에 device_id가 없습니다."
                )
                return

            print(
                f"어종 분석 요청 수신 - device_id: {device_id}"
            )

            # ==================================
            # 3초 어종 분석
            # ==================================
            result = self.species_service.analyze(
                self.camera
            )

            # ==================================
            # 분석 결과 Node로 전송
            # ==================================
            await self.send_json(
                {
                    "type": "species_result",
                    "device_id": device_id,
                    "data": result
                }
            )

        else:
            print(
                "알 수 없는 메시지 type:",
                message_type
            )

    # ==========================================
    # WebSocket 실행
    # 연결 끊기면 자동 재연결
    # ==========================================
    async def run(self):

        self.running = True

        while self.running:

            try:
                await self.connect()

                await self.receive_loop()

            except Exception as e:

                print(
                    "WebSocket 연결 오류:",
                    e
                )

            if self.running:

                print(
                    "3초 후 WebSocket 재연결..."
                )

                await asyncio.sleep(3)

    # ==========================================
    # 종료
    # ==========================================
    async def stop(self):

        self.running = False

        if self.websocket is not None:
            await self.websocket.close()

        print("WebSocket 종료")