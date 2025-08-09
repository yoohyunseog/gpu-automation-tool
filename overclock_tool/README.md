Windows CPU 오버클럭 보조 도구 (안전 지향)

이 도구는 저수준 MSR/드라이버 조작 없이, 벤더 제공 툴(Intel XTU, AMD Ryzen Master)과 Windows 전원 계획을 래핑하여 비교적 안전하게 오버클럭 워크플로우를 보조합니다.

중요: 하드웨어 손상/보증 무효/데이터 손실 위험이 있습니다. 충분한 지식, 냉각, 전원 환경, 단계적 테스트가 필수입니다. 모든 조작은 본인 책임입니다.

주요 기능
- CPU 벤더/정보 감지: `detect`
- 벤더 공식 툴 실행: `open-vendor Intel|AMD`
- Intel XTU 프로필 적용(.xtuprofile): `intel-apply-profile` (선택적 간단 스트레스 테스트)
- 전원 계획 전환: `set-powerplan balanced|high|ultimate`
- 간단 모니터링: `monitor [--watch --interval 2.0]`

요구 사항
- Windows 10/11
- 관리자 권한 PowerShell/CMD 권장
- Intel: Intel Extreme Tuning Utility(XTU) 설치 필요
- AMD: AMD Ryzen Master 설치 필요

설치 및 실행
프로젝트 루트 예시:

```powershell
# 파이썬으로 실행 (가상환경 권장)
python overclock_tool/main.py detect
python overclock_tool/main.py open-vendor Intel
python overclock_tool/main.py intel-apply-profile C:\path\to\my.xtuprofile --stress 60
python overclock_tool/main.py set-powerplan high
python overclock_tool/main.py monitor --watch --interval 1.5
```

사용 팁
- 먼저 BIOS에서 오버클럭 관련 옵션(전력/PL1/PL2, PBO 등)을 이해하고 필요 시 벤더 가이드를 따르세요.
- 프로필 적용 전, 동일 시스템/버전에 맞춘 XTU 프로필을 사용하세요.
- 안정성 검증은 벤더 툴의 스트레스 테스트, 추가로 Cinebench, AIDA64, OCCT 등으로 장시간 테스트하세요.
- 온도/전력/전압 한계를 넘지 않도록 모니터링하세요.

주의사항(중요)
- 이 스크립트는 직접적인 레지스터/드라이버 조작을 하지 않습니다. 공식 툴 호출만 돕습니다.
- 각 벤더 툴 버전에 따라 CLI 동작이 다를 수 있습니다. XTU CLI(XtuCli.exe) 경로 인식이 안 될 경우 수동으로 설치/경로 확인이 필요합니다.
- Ryzen Master는 명시적 CLI가 제한적입니다. 본 도구는 실행만 지원합니다.

문제 해결
- `powercfg /L`에서 전원 계획이 보이지 않으면 Windows 기능/정책을 확인하세요. Ultimate Performance는 워크스테이션/엔터프라이즈에서만 기본 제공될 수 있습니다.
- 관리자 권한으로 실행했는지 확인하세요.

라이선스
- 내부 도구/예시 용도. 하드웨어 리스크는 사용자 책임입니다.


