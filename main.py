import asyncio

from vision.rtsp_camera import RTSPCamera
from vision.detector import FishDetector
from services.species_service import SpeciesService
from websocket.ws_client import WebSocketClient


async def main():

    # ==========================================
    # RTSP 카메라 시작
    # ==========================================
    camera = RTSPCamera()
    camera.start()

    # ==========================================
    # YOLO + ByteTrack 모델 준비
    # ==========================================
    detector = FishDetector()

    # ==========================================
    # 어종 분석 서비스 준비
    # ==========================================
    species_service = SpeciesService(
        detector=detector,
        analysis_time=3.0
    )

    # ==========================================
    # WebSocket 클라이언트 준비
    # ==========================================
    ws_client = WebSocketClient(
        species_service=species_service,
        camera=camera
    )

    try:
        # Node.js WebSocket 서버에 연결
        # 이후 species 요청을 계속 기다림
        await ws_client.run()

    except KeyboardInterrupt:
        print("\nAI 서버 종료 요청")

    finally:
        # ======================================
        # 종료 처리
        # ======================================
        camera.stop()

        await ws_client.stop()

        print("AI 서버 종료 완료")


if __name__ == "__main__":
    asyncio.run(main())