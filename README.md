# 관상어용 스마트 어항 시스템 AI Backend

YOLO11n 기반 관상어 어종 및 개체 수 분석과 SOM 기반 활동성 이상 탐지를 수행하는 FastAPI 서버입니다.

## 📁 프로젝트 구조

```text
fish-ai-server/
├── core/
│   └── yolo_priority.py            # YOLO 분석 우선순위 제어
├── services/
│   ├── activity_result_service.py  # 활동성 분석 결과 처리
│   ├── activity_service.py         # SOM 기반 활동성 이상 탐지
│   └── species_service.py          # 어종 및 개체 수 분석
├── trackers/
│   └── custom_bytetrack.yaml       # ByteTrack 추적 설정
├── vision/
│   ├── activity_visualizer.py      # 활동성 추적 결과 시각화
│   ├── detector_csrt.py            # YOLO + CSRT 기반 물고기 추적
│   ├── detector.py                 # YOLO 기반 객체 탐지
│   └── rtsp_camera.py              # RTSP 실시간 영상 수신
├── websocket/
│   └── ws_client.py                # Node.js 서버 WebSocket 통신
├── weights/
│   ├── best.pt                     # 학습된 YOLO11n 모델
│   └── som_model.npz               # 학습된 SOM 모델
├── .env.example                    # 환경 변수 예시
├── .gitignore                      # Git 제외 설정
├── config.py                       # AI 서버 환경 설정
├── main.py                         # FastAPI 서버 실행 및 초기화
├── requirements.txt                # Python 패키지 목록
└── README.md                       # 프로젝트 설명 및 실행 방법
```

## 🚀 실행 방법

### 1️⃣ 프로젝트 클론

```bash
git clone https://github.com/dakyung427/fish-ai-server.git
cd fish-ai-server
```

### 2️⃣ 가상환경 생성 및 활성화

Python 3.11 및 Windows PowerShell 환경을 기준으로 합니다.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3️⃣ 패키지 설치

```powershell
pip install -r requirements.txt
```

### 4️⃣ 환경 변수 설정

프로젝트 루트 디렉터리에 `.env` 파일을 생성합니다.

```powershell
Copy-Item .env.example .env
```

`.env` 파일에 다음 항목을 설정합니다.

```dotenv
RTSP_URL=rtsp://<카메라서버주소>:8554/cam
NODE_WS_URL=ws://<백엔드서버주소>:3000
YOLO_MODEL_PATH=weights/best.pt
BYTETRACK_CONFIG=trackers/custom_bytetrack.yaml
```

- `RTSP_URL`: Raspberry Pi에서 송출하는 실시간 영상 주소
- `NODE_WS_URL`: Node.js 백엔드 WebSocket 서버 주소
- `YOLO_MODEL_PATH`: 학습된 YOLO 모델 경로
- `BYTETRACK_CONFIG`: ByteTrack 설정 파일 경로

> 위 환경 변수 이름은 `config.py`와 `.env.example`의 실제 설정과 일치해야 합니다. `main.py`에서는 활동성 분석용 모델을 `weights/best.pt`, SOM 모델을 `weights/som_model.npz`에서 불러옵니다.

### 5️⃣ AI 서버 실행

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

서버 실행 시 RTSP 영상 연결, YOLO 모델 초기화, SOM 활동성 분석 및 WebSocket 통신이 시작됩니다.

- 서버 상태 확인: `http://localhost:8000/health`
- 활동성 추적 영상 확인: `http://localhost:8000/activity`

## 🧠 AI 분석 모델

### 1️⃣ 어종 및 개체 수 분석 (YOLO11n + ByteTrack)

- YOLO11n을 활용한 관상어 객체 탐지 및 어종 분류
- 13종 관상어 데이터셋 기반 모델 학습
- ByteTrack을 활용한 객체 추적 및 개체 수 분석
- 분석 결과를 WebSocket을 통해 백엔드 서버로 전송

### 2️⃣ 활동성 이상 탐지 (YOLO11n + CSRT + SOM)

- YOLO11n 기반 물고기 초기 위치 탐지
- CSRT를 활용한 실시간 객체 추적
- HSV 색상 히스토그램을 활용한 추적 결과 검증
- 주기적인 YOLO 재탐지를 통한 추적 보정
- 이동 거리, 이동 속도, 정지 비율 등의 활동성 특징 추출
- SOM(Self-Organizing Map) 기반 활동성 이상 탐지
- 연속적인 이상 탐지 결과를 활용한 최종 상태 판단

## 🔗 시스템 연동

```text
Raspberry Pi Camera
        │
        │ RTSP
        ▼
    AI Backend
        │
        ├── YOLO11n + ByteTrack
        │       └── 어종 및 개체 수 분석
        │
        └── YOLO11n + CSRT + SOM
                └── 활동성 이상 탐지
        │
        │ WebSocket
        ▼
   Node.js Backend
        │
        ▼
   Flutter Frontend
```

## 🛠 기술 스택

- **Language:** Python 3.11
- **Framework:** FastAPI
- **Object Detection:** YOLO11n
- **Object Tracking:** ByteTrack, OpenCV CSRT
- **Anomaly Detection:** Self-Organizing Map (SOM)
- **Video Processing:** OpenCV, RTSP
- **Communication:** WebSocket

## 📊 YOLO 모델 성능

YOLO11n 모델의 검증 데이터셋 평가 결과입니다.

| 평가 지표 | 결과 |
|---|---:|
| Precision | 91.9% |
| Recall | 91.5% |
| mAP@0.5 | 94.1% |
| mAP@0.5:0.95 | 73.0% |
