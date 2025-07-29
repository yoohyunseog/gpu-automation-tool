# 🚀 gpu-automation-tool

GPU 프로세스 자동화 및 모니터링 도구

## 📋 기능

- 🖥️ GPU 사용률 모니터링
- ⚡ 프로세스 GPU 최적화  
- 🔄 자동 GPU 작업 생성
- 📊 실시간 성능 추적
- 🎯 멀티 GPU 지원
- 🔧 설정 관리

## 🛠️ 설치

### 요구사항
- Python 3.8+
- NVIDIA GPU (CUDA 지원)
- Windows 10/11

### 설치 방법

```bash
# 저장소 클론
git clone https://github.com/yourusername/gpu-automation-tool.git
cd gpu-automation-tool

# 가상환경 생성
python -m venv venv
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 🚀 사용법

### 기본 사용
```python
from src.gpu_automation.core import GPUProcessAutomation

# GPU 자동화 객체 생성
automation = GPUProcessAutomation()

# 프로세스 최적화
automation.optimize_process_for_gpu("python.exe", mode="normal")
```

### 고급 사용
```python
# 최대 GPU 자원 할당
automation.optimize_process_for_gpu("python.exe", mode="maximum")

# 특정 GPU 사용
automation.optimize_process_for_gpu("python.exe", mode="gpu_0", gpu_id=0)

# 멀티 GPU 분산
automation.optimize_process_for_gpu("python.exe", mode="multi_gpu")
```

## 📁 프로젝트 구조

```
gpu-automation-tool/
├── src/
│   ├── gpu_automation/    # 핵심 GPU 자동화 모듈
│   ├── utils/            # 유틸리티 함수들
│   └── config/           # 설정 관리
├── tests/                # 테스트 코드
├── docs/                 # 문서
├── scripts/              # 실행 스크립트
├── data/                 # 데이터 및 로그
└── examples/             # 사용 예제
```

## 🔧 설정

`config.json` 파일에서 설정을 관리합니다:

```json
{
    "gpu_settings": {
        "default_mode": "normal",
        "max_memory_usage": 0.9,
        "monitoring_interval": 1.0
    },
    "process_config": {
        "python.exe": {
            "priority": "high",
            "mode": "normal",
            "gpu_id": 0
        }
    }
}
```

## 📊 모니터링

실시간 GPU 사용률 모니터링:

```python
# GPU 사용률 확인
automation.monitor_gpu_usage()

# 특정 GPU 모니터링
automation.monitor_specific_gpu_usage(0)
```

## 🐛 문제 해결

### 일반적인 문제들

1. **CUDA 오류**
   - NVIDIA 드라이버 업데이트
   - PyTorch CUDA 버전 확인

2. **권한 오류**
   - 관리자 권한으로 실행
   - Windows Defender 예외 설정

3. **GPU 인식 안됨**
   - nvidia-smi 명령어로 GPU 확인
   - CUDA 설치 확인

## 🤝 기여

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👨‍💻 작성자

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!
