import asyncio

from config import (
    USE_TEST_VIDEO,
    TEST_VIDEO_PATH
)

from vision.rtsp_camera import RTSPCamera
from vision.video_camera import VideoCamera
from vision.detector import FishDetector

from services.species_service import SpeciesService

from websocket.ws_client import WebSocketClient


async def main():

    camera = None
    ws_client = None

    try:

        # ==========================================
        # 영상 입력 선택
        # ==========================================

        if USE_TEST_VIDEO:

            print("================================")
            print("MP4 테스트 영상 모드")
            print("================================")

            camera = VideoCamera(
                TEST_VIDEO_PATH
            )

            camera.start()

        else:

            print("================================")
            print("RTSP 실시간 영상 모드")
            print("================================")

            camera = RTSPCamera()

            camera.start()


        # ==========================================
        # YOLO + ByteTrack
        # ==========================================

        detector = FishDetector()


        # ==========================================
        # 어종 분석 서비스
        # ==========================================

        species_service = SpeciesService(
            detector=detector,
            analysis_time=3.0
        )


        # ==========================================
        # WebSocket 클라이언트
        # ==========================================

        ws_client = WebSocketClient(
            species_service=species_service,
            camera=camera
        )


        # ==========================================
        # Node.js WebSocket 서버 연결
        # ==========================================

        await ws_client.run()


    finally:

        # ==========================================
        # 종료 처리
        # ==========================================

        if camera is not None:
            camera.stop()

        if ws_client is not None:
            await ws_client.stop()

        print("AI 서버 종료 완료")


if __name__ == "__main__":
    asyncio.run(main())