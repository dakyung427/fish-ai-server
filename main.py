import asyncio

from vision.rtsp_camera import RTSPCamera
from vision.detector import FishDetector
from services.species_service import SpeciesService
from websocket.ws_client import WebSocketClient


async def main():

    # =========================
    # RTSP 카메라 시작
    # =========================
    camera = RTSPCamera()
    camera.start()


    # =========================
    # YOLO + ByteTrack
    # =========================
    detector = FishDetector()


    # =========================
    # 어종 분석 서비스
    # =========================
    species_service = SpeciesService(
        detector=detector,
        analysis_time=3.0
    )


    # =========================
    # WebSocket 클라이언트
    # =========================
    ws_client = WebSocketClient(
        species_service=species_service,
        camera=camera
    )


    try:

        # Node.js WebSocket 서버 연결
        await ws_client.run()

    finally:

        # 프로그램 종료 시 정리
        camera.stop()

        await ws_client.stop()


if __name__ == "__main__":
    asyncio.run(main())