import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
import ctypes


def ensure_windows() -> None:
    if os.name != "nt":
        print("이 프로그램은 Windows에서만 동작합니다.")
        sys.exit(1)


def run_command(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check, shell=False)


def detect_cpu_vendor() -> str:
    try:
        result = run_command(["wmic", "cpu", "get", "Manufacturer,Name", "/value"], check=True)
        text = result.stdout
        if "GenuineIntel" in text or "Intel" in text:
            return "Intel"
        if "AuthenticAMD" in text or "AMD" in text:
            return "AMD"
    except Exception:
        pass
    # Fallback using environment or platform string
    import platform

    plat = platform.platform()
    if "AMD" in plat:
        return "AMD"
    if "Intel" in plat:
        return "Intel"
    return "Unknown"


def get_cpu_info() -> dict:
    info: dict[str, str | int] = {"name": "", "manufacturer": "", "current_mhz": None, "max_mhz": None,
                                   "logical_processors": None, "physical_cores": None}
    try:
        result = run_command(["wmic", "cpu", "get", "Name,Manufacturer,CurrentClockSpeed,MaxClockSpeed,NumberOfCores,NumberOfLogicalProcessors", "/format:list"], check=True)
        text = result.stdout
        for line in text.splitlines():
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "Name":
                info["name"] = value
            elif key == "Manufacturer":
                info["manufacturer"] = value
            elif key == "CurrentClockSpeed":
                info["current_mhz"] = int(value) if value.isdigit() else None
            elif key == "MaxClockSpeed":
                info["max_mhz"] = int(value) if value.isdigit() else None
            elif key == "NumberOfCores":
                try:
                    info["physical_cores"] = int(value)
                except Exception:
                    info["physical_cores"] = None
            elif key == "NumberOfLogicalProcessors":
                try:
                    info["logical_processors"] = int(value)
                except Exception:
                    info["logical_processors"] = None
    except Exception:
        pass
    return info


def search_in_program_files(candidate_names: list[str]) -> str | None:
    candidates: list[Path] = []
    program_files = [os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432")]
    for root in [p for p in program_files if p]:
        for name in candidate_names:
            # Quick known subpaths first
            known_subpaths = [
                Path(root) / name,
            ]
            for p in known_subpaths:
                if p.exists():
                    return str(p)
            # Fallback: shallow search in top-level vendor dirs
            vendors = ["Intel", "Intel(R) Extreme Tuning Utility", "AMD", "RyzenMaster", "AMD Ryzen Master"]
            for vendor in vendors:
                base = Path(root) / vendor
                if base.exists():
                    for path in base.rglob(Path(name).name):
                        return str(path)
    # PATH lookup
    for name in candidate_names:
        found = shutil.which(name)
        if found:
            return found
    return None


def find_xtu_cli() -> str | None:
    # Common paths for Intel XTU CLI
    names = [
        r"C:\\Program Files (x86)\\Intel\\Intel(R) Extreme Tuning Utility\\Client\\XtuCli.exe",
        r"C:\\Program Files\\Intel\\Intel(R) Extreme Tuning Utility\\Client\\XtuCli.exe",
        "XtuCli.exe",
    ]
    return search_in_program_files(names)


def find_ryzen_master() -> str | None:
    names = [
        r"C:\\Program Files\\AMD\\RyzenMaster\\bin\\AMDRyzenMaster.exe",
        r"C:\\Program Files\\AMD\\RyzenMaster\\bin\\RyzenMaster.exe",
        r"C:\\Program Files\\AMD\\RyzenMaster\\AMDRyzenMaster.exe",
        "AMDRyzenMaster.exe",
        "RyzenMaster.exe",
    ]
    return search_in_program_files(names)


def require_admin_note() -> None:
    print("관리자 권한으로 PowerShell 또는 CMD를 실행한 뒤 이 프로그램을 실행하는 것을 권장합니다.")


def intel_apply_profile(profile_path: str, stress_seconds: int | None = None) -> None:
    xtu = find_xtu_cli()
    if not xtu:
        print("Intel XTU CLI(XtuCli.exe)를 찾지 못했습니다. Intel XTU를 설치한 뒤 다시 시도하세요.")
        sys.exit(1)
    prof = Path(profile_path)
    if not prof.exists():
        print(f"프로필 파일을 찾을 수 없습니다: {prof}")
        sys.exit(1)
    print(f"XTU CLI: {xtu}")
    print(f"프로필 적용: {prof}")
    # Import/apply profile
    try:
        res = run_command([xtu, "-i", str(prof)])
        print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print("XTU 프로필 적용 실패:")
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

    if stress_seconds and stress_seconds > 0:
        print(f"간단한 안정성 테스트 실행 ({stress_seconds}초)...")
        try:
            res = run_command([xtu, "-t", "-duration", str(stress_seconds)])
            print(res.stdout)
            if res.stderr:
                print(res.stderr, file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print("XTU 스트레스 테스트 실패(무시 가능):")
            print(e.stdout)
            print(e.stderr, file=sys.stderr)


def open_vendor_tool(vendor: str) -> None:
    if vendor == "Intel":
        path = find_xtu_cli()
        if path:
            # XtuCli는 CLI이므로 GUI는 XTU 메인 앱을 열도록 유도
            gui_candidates = [
                r"C:\\Program Files (x86)\\Intel\\Intel(R) Extreme Tuning Utility\\Client\\Intel XTU.exe",
                r"C:\\Program Files\\Intel\\Intel(R) Extreme Tuning Utility\\Client\\Intel XTU.exe",
            ]
            gui = search_in_program_files(gui_candidates)
            exe = gui or path
            print(f"실행: {exe}")
            os.startfile(exe)
            return
        print("Intel XTU를 찾지 못했습니다. 설치 후 다시 시도하세요.")
        sys.exit(1)
    elif vendor == "AMD":
        path = find_ryzen_master()
        if path:
            print(f"실행: {path}")
            os.startfile(path)
            return
        print("AMD Ryzen Master를 찾지 못했습니다. 설치 후 다시 시도하세요.")
        sys.exit(1)
    else:
        print("지원되지 않는 또는 알 수 없는 벤더입니다.")
        sys.exit(1)


def set_power_plan(mode: str) -> None:
    # mode: balanced, high, ultimate
    print("전원 관리 옵션 검색 중...")
    try:
        res = run_command(["powercfg", "/L"])
    except subprocess.CalledProcessError as e:
        print("powercfg 실행 실패. 관리자 권한이 필요할 수 있습니다.")
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    text = res.stdout
    # Extract GUID and name lines
    entries = []
    for line in text.splitlines():
        m = re.search(r"([A-F0-9\-]{36}).*?\((.*?)\)", line, re.IGNORECASE)
        if m:
            entries.append((m.group(1), m.group(2)))
    target_name = {
        "balanced": "Balanced",
        "high": "High performance",
        "ultimate": "Ultimate Performance",
    }.get(mode.lower())
    if not target_name:
        print("mode는 balanced|high|ultimate 중 하나여야 합니다.")
        sys.exit(1)
    # Try to find by case-insensitive contains
    guid = None
    for g, name in entries:
        if target_name.lower() in name.lower():
            guid = g
            break
    if not guid:
        # 이름 기반 탐색 실패 시, 표준 GUID로 직접 활성화 시도
        guid_fallback = None
        if mode.lower() == "ultimate":
            guid_fallback = GUID_ULTIMATE_SCHEME
        elif mode.lower() == "high":
            guid_fallback = GUID_HIGH_PERF
        elif mode.lower() == "balanced":
            guid_fallback = GUID_BALANCED
        if guid_fallback:
            try:
                run_command(["powercfg", "/S", guid_fallback])
                print(f"전원 계획 전환 완료(직접 GUID): {guid_fallback}")
                return
            except subprocess.CalledProcessError:
                try:
                    run_command(["powercfg", "-duplicatescheme", guid_fallback], check=False)
                    run_command(["powercfg", "/S", guid_fallback])
                    print(f"전원 계획 생성/전환 완료(직접 GUID): {guid_fallback}")
                    return
                except subprocess.CalledProcessError as e:
                    print("전원 계획 전환 실패(직접 GUID)")
                    print(e.stderr, file=sys.stderr)
        print(f"전원 계획 '{target_name}'을 찾지/활성화하지 못했습니다. 수동 확인이 필요할 수 있습니다.")
        print("현재 전원 계획 목록:")
        for g, name in entries:
            print(f"- {name}: {g}")
        sys.exit(1)
    try:
        run_command(["powercfg", "/S", guid])
        print(f"전원 계획 전환 완료: {target_name} ({guid})")
    except subprocess.CalledProcessError as e:
        print("전원 계획 전환 실패. 관리자 권한이 필요할 수 있습니다.")
        print(e.stderr, file=sys.stderr)
        sys.exit(1)


def monitor_cpu(watch: bool, interval_sec: float) -> None:
    def one_shot() -> None:
        info = get_cpu_info()
        print("CPU 정보:")
        print(f"- 제조사: {info.get('manufacturer')}")
        print(f"- 모델: {info.get('name')}")
        print(f"- 물리 코어: {info.get('physical_cores')}")
        print(f"- 논리 스레드: {info.get('logical_processors')}")
        cur = info.get("current_mhz")
        mx = info.get("max_mhz")
        if cur:
            print(f"- 현재 클럭: {cur} MHz")
        if mx:
            print(f"- 명시된 최대 클럭: {mx} MHz")

    if not watch:
        one_shot()
        return
    try:
        while True:
            os.system("cls")
            one_shot()
            print(f"\n{interval_sec}초 후 갱신...")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        pass


SUB_PROCESSOR = "54533251-82be-4824-96c1-47b60b740d00"
PROC_MIN = "893dee8e-2bef-41e0-89c6-b55d0929964c"
PROC_MAX = "bc5038f7-23e0-4960-96da-33abaf5935ec"
GUID_ULTIMATE_SCHEME = "e9a42b02-d5df-448d-aa00-03f14749eb61"
GUID_HIGH_PERF = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
GUID_BALANCED = "381b4222-f694-41f0-9685-ff5bb260df2e"


def list_power_plans() -> list[tuple[str, str, bool]]:
    try:
        res = run_command(["powercfg", "/L"])
    except subprocess.CalledProcessError as e:
        print("powercfg 실행 실패. 관리자 권한이 필요할 수 있습니다.")
        print(e.stderr, file=sys.stderr)
        return []
    plans: list[tuple[str, str, bool]] = []
    for line in res.stdout.splitlines():
        # Example:  Power Scheme GUID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  (Balanced) *
        m = re.search(r"([A-F0-9\-]{36}).*?\((.*?)\).*?(\*)?", line, re.IGNORECASE)
        if m:
            guid = m.group(1)
            name = m.group(2)
            active = bool(m.group(3))
            plans.append((guid, name, active))
    return plans


def ensure_plan_active(preferred: str) -> str | None:
    # Attempt to activate by canonical GUIDs first (locale independent)
    order: list[str] = []
    pref = preferred.lower()
    if pref == "ultimate":
        order = [GUID_ULTIMATE_SCHEME, GUID_HIGH_PERF, GUID_BALANCED]
    elif pref == "high":
        order = [GUID_HIGH_PERF, GUID_BALANCED]
    else:
        order = [GUID_BALANCED, GUID_HIGH_PERF]

    def try_activate(guid: str) -> bool:
        try:
            run_command(["powercfg", "/S", guid])
            print(f"전원 계획 활성화: {guid}")
            return True
        except subprocess.CalledProcessError:
            try:
                run_command(["powercfg", "-duplicatescheme", guid], check=False)
                run_command(["powercfg", "/S", guid])
                print(f"전원 계획 생성/활성화: {guid}")
                return True
            except subprocess.CalledProcessError:
                return False

    for guid in order:
        if try_activate(guid):
            return guid

    # As a last resort, fallback to name matching (for custom plans)
    plans = list_power_plans()
    target_name = {
        "ultimate": "Ultimate Performance",
        "high": "High performance",
        "balanced": "Balanced",
    }.get(pref, "Ultimate Performance")
    for guid, name, _ in plans:
        if target_name.lower() in name.lower():
            try:
                run_command(["powercfg", "/S", guid])
                print(f"전원 계획 활성화(이름): {guid}")
                return guid
            except subprocess.CalledProcessError:
                continue
    print("적절한 전원 계획을 찾지/활성화하지 못했습니다.")
    return None


def set_processor_100_percent(scheme_guid: str) -> None:
    try:
        # AC
        run_command(["powercfg", "/SETACVALUEINDEX", scheme_guid, SUB_PROCESSOR, PROC_MIN, "100"])  # Min 100
        run_command(["powercfg", "/SETACVALUEINDEX", scheme_guid, SUB_PROCESSOR, PROC_MAX, "100"])  # Max 100
        # DC
        run_command(["powercfg", "/SETDCVALUEINDEX", scheme_guid, SUB_PROCESSOR, PROC_MIN, "100"])  # Min 100
        run_command(["powercfg", "/SETDCVALUEINDEX", scheme_guid, SUB_PROCESSOR, PROC_MAX, "100"])  # Max 100
        run_command(["powercfg", "/S", scheme_guid])
        print("프로세서 최소/최대 상태 100%로 설정 완료(AC/DC)")
    except subprocess.CalledProcessError as e:
        print("프로세서 전원 인덱스 설정 실패. 관리자 권한이 필요할 수 있습니다.")
        print(e.stderr, file=sys.stderr)


def prevent_sleep_tick(include_display: bool) -> None:
    # Use SetThreadExecutionState to keep system awake
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if include_display:
        flags |= ES_DISPLAY_REQUIRED
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def always_max(preferred_plan: str, once: bool, interval_sec: float, include_display: bool) -> None:
    guid = ensure_plan_active(preferred_plan)
    if not guid:
        return
    set_processor_100_percent(guid)
    if once:
        return
    try:
        while True:
            prevent_sleep_tick(include_display)
            # Re-assert settings
            ensure_plan_active(preferred_plan)
            set_processor_100_percent(guid)
            print(f"최고 성능 유지 중... {interval_sec}초 후 재확인")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        pass


def main() -> None:
    ensure_windows()
    parser = argparse.ArgumentParser(
        description=(
            "안전 지향의 CPU 오버클럭 보조 도구.\n"
            "- Intel: Intel XTU의 CLI(XtuCli.exe)를 통해 미리 만든 프로필(.xtuprofile)을 적용\n"
            "- AMD: AMD Ryzen Master 앱 실행을 도와주고, 전원 계획 전환/모니터링 제공\n\n"
            "항상 관리자 권한으로 실행하고, 충분한 쿨링/전원 환경에서 신중히 작업하세요."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "예시:\n"
            "  python overclock_tool/main.py detect\n"
            "  python overclock_tool/main.py open-vendor Intel\n"
            "  python overclock_tool/main.py intel-apply-profile C:\\path\\to\\my.xtuprofile --stress 60\n"
            "  python overclock_tool/main.py set-powerplan high\n"
            "  python overclock_tool/main.py monitor --watch --interval 1.5\n"
        ),
    )

    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("detect", help="CPU 벤더 및 기본 정보 표시")

    p_open = sub.add_parser("open-vendor", help="벤더 공식 툴 열기 (Intel XTU / AMD Ryzen Master)")
    p_open.add_argument("vendor", choices=["Intel", "AMD"], help="CPU 벤더")

    p_apply = sub.add_parser("intel-apply-profile", help="Intel XTU 프로필(.xtuprofile) 적용")
    p_apply.add_argument("profile", help=".xtuprofile 파일 경로")
    p_apply.add_argument("--stress", type=int, default=0, help="적용 후 간단 스트레스 테스트(초), 0은 생략")

    p_pp = sub.add_parser("set-powerplan", help="전원 계획 전환 (오버클럭 시 High/Ultimate 권장)")
    p_pp.add_argument("mode", choices=["balanced", "high", "ultimate"], help="전환할 전원 계획")

    p_mon = sub.add_parser("monitor", help="간단 모니터링(클럭 등)" )
    p_mon.add_argument("--watch", action="store_true", help="주기적으로 갱신")
    p_mon.add_argument("--interval", type=float, default=2.0, help="갱신 주기(초)")

    p_max = sub.add_parser("always-max", help="전원 계획/프로세서 상태를 강제로 최고 성능으로 유지")
    p_max.add_argument("--plan", choices=["ultimate", "high", "balanced"], default="ultimate", help="선호 전원 계획(미존재 시 대체)")
    p_max.add_argument("--once", action="store_true", help="1회 적용 후 종료")
    p_max.add_argument("--interval", type=float, default=30.0, help="재적용/검사 주기(초)")
    p_max.add_argument("--include-display", action="store_true", help="디스플레이 절전도 방지")

    p_auto = sub.add_parser("auto", help="자동 최적화: 전원계획 설정→프로세서 100%%→(옵션)벤더툴 실행/XTU프로필/모니터링/항상최고유지")
    p_auto.add_argument("--plan", choices=["ultimate", "high", "balanced"], default="ultimate", help="선호 전원 계획")
    p_auto.add_argument("--no-open-vendor", action="store_true", help="벤더 툴 실행 생략")
    p_auto.add_argument("--intel-profile", type=str, default="", help="인텔일 때 적용할 XTU 프로필 경로(.xtuprofile)")
    p_auto.add_argument("--stress", type=int, default=0, help="XTU 프로필 적용 후 간단 스트레스 테스트(초)")
    p_auto.add_argument("--monitor", action="store_true", help="간단 모니터링 1회 표시")
    p_auto.add_argument("--keep", action="store_true", help="항상 최고 성능 유지 루프 실행")
    p_auto.add_argument("--keep-interval", type=float, default=30.0, help="유지 루프 재적용 주기(초)")
    p_auto.add_argument("--include-display", action="store_true", help="유지 루프에서 디스플레이 절전도 방지")

    args = parser.parse_args()
    require_admin_note()

    if args.cmd is None:
        parser.print_help()
        return

    if args.cmd == "detect":
        vendor = detect_cpu_vendor()
        info = get_cpu_info()
        print(f"벤더: {vendor}")
        print(f"모델: {info.get('name')}")
        print(f"제조사 문자열: {info.get('manufacturer')}")
        print(f"현재/최대 클럭(MHz): {info.get('current_mhz')} / {info.get('max_mhz')}")
        print(f"물리 코어 / 논리 스레드: {info.get('physical_cores')} / {info.get('logical_processors')}")
        if vendor == "Intel":
            print(f"XTU CLI: {find_xtu_cli() or '미설치/미발견'}")
        elif vendor == "AMD":
            print(f"Ryzen Master: {find_ryzen_master() or '미설치/미발견'}")
        return

    if args.cmd == "open-vendor":
        open_vendor_tool(args.vendor)
        return

    if args.cmd == "intel-apply-profile":
        intel_apply_profile(args.profile, stress_seconds=int(args.stress))
        return

    if args.cmd == "set-powerplan":
        set_power_plan(args.mode)
        return

    if args.cmd == "monitor":
        monitor_cpu(args.watch, args.interval)
        return

    if args.cmd == "always-max":
        always_max(args.plan, args.once, args.interval, args.include_display)
        return

    if args.cmd == "auto":
        vendor = detect_cpu_vendor()
        print(f"감지된 벤더: {vendor}")
        guid = ensure_plan_active(args.plan)
        if guid:
            set_processor_100_percent(guid)
        # Intel profile apply if requested and vendor is Intel
        if vendor == "Intel" and args.intel_profile:
            intel_apply_profile(args.intel_profile, stress_seconds=int(args.stress))
        # Open vendor tool unless disabled
        if not args.no_open_vendor:
            if vendor in ("Intel", "AMD"):
                try:
                    open_vendor_tool(vendor)
                except SystemExit:
                    # Keep flow even if vendor tool missing
                    pass
        # One-shot monitor
        if args.monitor:
            monitor_cpu(False, 0)
        # Keep always-max loop
        if args.keep:
            always_max(args.plan, False, args.keep_interval, args.include_display)
        return


if __name__ == "__main__":
    main()


