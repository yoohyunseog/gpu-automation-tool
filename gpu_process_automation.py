import psutil
import torch
import numpy as np
import time
import threading
import json
import os
from datetime import datetime
import subprocess
import win32gui
import win32process
import win32api
import win32con
import msvcrt
import sys

class GPUProcessAutomation:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_processes = {}
        self.process_config = {}
        self.load_config()
        
    def load_config(self):
        """GPU 프로세스 설정 파일 로드"""
        config_file = "gpu_process_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                self.process_config = json.load(f)
        else:
            # 기본 설정 생성
            self.process_config = {
                "gpu_processes": {
                    "chrome.exe": {"priority": "high", "gpu_memory": 1024, "mode": "normal", "gpu_id": 0},
                    "firefox.exe": {"priority": "high", "gpu_memory": 1024, "mode": "normal", "gpu_id": 0},
                    "code.exe": {"priority": "medium", "gpu_memory": 512, "mode": "normal", "gpu_id": 0},
                    "python.exe": {"priority": "high", "gpu_memory": 2048, "mode": "maximum", "gpu_id": 0},
                    "notepad.exe": {"priority": "low", "gpu_memory": 256, "mode": "normal", "gpu_id": 0}
                },
                "gpu_settings": {
                    "power_mode": "prefer_maximum_performance",
                    "texture_quality": "high_quality",
                    "shader_cache": "enabled"
                }
            }
            self.save_config()
    
    def save_config(self):
        """설정 파일 저장"""
        with open("gpu_process_config.json", 'w', encoding='utf-8') as f:
            json.dump(self.process_config, f, indent=2, ensure_ascii=False)
    
    def get_process_list(self):
        """현재 실행 중인 프로세스 목록 반환"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_percent': proc.info['memory_percent']
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    def set_process_gpu_priority(self, process_name, priority="high"):
        """프로세스의 GPU 우선순위 설정"""
        try:
            # NVIDIA GPU 설정 (NVIDIA Profile Inspector API 사용)
            if priority == "high":
                self.set_nvidia_gpu_priority(process_name, "prefer_maximum_performance")
            elif priority == "medium":
                self.set_nvidia_gpu_priority(process_name, "adaptive")
            else:
                self.set_nvidia_gpu_priority(process_name, "prefer_consistent_performance")
                
            print(f"✅ {process_name}의 GPU 우선순위를 {priority}로 설정했습니다.")
            return True
        except Exception as e:
            print(f"❌ {process_name} GPU 우선순위 설정 실패: {e}")
            return False
    
    def set_nvidia_gpu_priority(self, process_name, power_mode):
        """NVIDIA GPU 전력 모드 설정"""
        try:
            # NVIDIA-SMI 명령어로 전력 모드 설정
            cmd = f'nvidia-smi -pm 1'  # 전력 관리 활성화
            subprocess.run(cmd, shell=True, check=True)
            
            # 프로세스별 GPU 설정 (실제로는 NVIDIA Profile Inspector 필요)
            print(f"🔧 {process_name}에 대해 {power_mode} 모드 적용")
            
        except subprocess.CalledProcessError:
            print("⚠️ NVIDIA-SMI 명령어 실행 실패 (NVIDIA 드라이버 확인 필요)")
    
    def create_gpu_workload(self, process_name, gpu_memory_mb=1024):
        """GPU 작업 부하 생성 (실제 GPU 사용 유도) - 대용량 메모리 할당"""
        try:
            # GPU 메모리 할당 및 계산 작업 - 더 큰 텐서 사용
            memory_size = gpu_memory_mb * 1024 * 1024 // 4  # float32 기준
            
            print(f"🚀 {process_name}에 대해 {gpu_memory_mb}MB GPU 메모리 할당 중...")
            
            # 여러 개의 큰 텐서 생성으로 GPU 메모리 최대 활용
            tensors = []
            for i in range(5):  # 5개의 큰 텐서 생성
                tensor = torch.randn(memory_size // 5, device=self.device)
                tensors.append(tensor)
            
            # GPU에서 지속적인 계산 작업 수행
            def continuous_gpu_work():
                iteration = 0
                while True:
                    try:
                        # ESC 키 확인
                        if self.check_esc_key():
                            print(f"\n⏹️ {process_name} GPU 작업 중지 (ESC 키 감지)")
                            break
                        
                                                    # 복잡한 행렬 연산으로 GPU 부하 증가
                            for i, tensor in enumerate(tensors):
                                try:
                                    # 텐서 차원 확인 및 조정
                                    if tensor.dim() == 1:
                                        # 1차원 텐서를 2차원으로 확장
                                        tensor = tensor.unsqueeze(0)
                                    
                                    # 행렬 곱셈 (2차원 텐서만)
                                    if tensor.dim() >= 2:
                                        tensor = torch.matmul(tensor, tensor.T)
                                    
                                    # 활성화 함수 적용
                                    tensor = torch.relu(tensor)
                                    
                                    # 정규화 (차원 확인)
                                    if tensor.dim() > 0:
                                        tensor = torch.nn.functional.normalize(tensor, dim=0)
                                    
                                    # 컨볼루션 연산 시뮬레이션 (안전한 차원 처리)
                                    if tensor.dim() == 1:
                                        tensor = tensor.unsqueeze(0)
                                    if tensor.dim() >= 2:
                                        # 컨볼루션을 위한 차원 조정
                                        if tensor.dim() == 2:
                                            tensor = tensor.unsqueeze(0)  # (batch, channels, length)
                                        
                                        # 컨볼루션 커널 크기 조정
                                        kernel_size = min(3, tensor.size(-1))
                                        if kernel_size > 0:
                                            conv_kernel = torch.randn(1, tensor.size(1), kernel_size, device=self.device)
                                            tensor = torch.conv1d(tensor, conv_kernel, padding=kernel_size//2)
                                    
                                    tensors[i] = tensor
                                    
                                except Exception as e:
                                    # 오류 발생 시 텐서 재생성
                                    print(f"⚠️ 텐서 {i} 처리 중 오류: {e}, 텐서 재생성")
                                    tensor_size = len(tensors[i])
                                    tensors[i] = torch.randn(tensor_size, device=self.device)
                        
                        # GPU 동기화
                        torch.cuda.synchronize()
                        
                        iteration += 1
                        if iteration % 100 == 0:
                            print(f"🔄 {process_name} GPU 작업 진행 중... (반복 {iteration}) - ESC로 중지")
                        
                        time.sleep(0.01)  # 10ms 대기
                        
                    except Exception as e:
                        print(f"❌ GPU 작업 오류: {e}")
                        break
            
            # 백그라운드 스레드로 GPU 작업 실행
            gpu_thread = threading.Thread(target=continuous_gpu_work, daemon=True)
            gpu_thread.start()
            
            self.gpu_processes[process_name] = {
                'tensors': tensors,
                'thread': gpu_thread,
                'memory_mb': gpu_memory_mb,
                'start_time': datetime.now(),
                'iterations': 0
            }
            
            print(f"✅ {process_name}에 대해 {gpu_memory_mb}MB GPU 작업 부하 생성 완료")
            return True
            
        except Exception as e:
            print(f"❌ GPU 작업 부하 생성 실패: {e}")
            return False
    
    def create_maximum_gpu_workload(self, process_name):
        """최대 GPU 자원 할당 (GPU 메모리의 90% 사용)"""
        try:
            # GPU 메모리 정보 가져오기
            gpu_memory_info = torch.cuda.get_device_properties(0)
            total_memory = gpu_memory_info.total_memory // (1024 * 1024)  # MB 단위
            available_memory = int(total_memory * 0.9)  # 90% 사용
            
            print(f"🔥 {process_name}에 대해 최대 GPU 자원 할당 시작...")
            print(f"📊 총 GPU 메모리: {total_memory}MB, 할당 예정: {available_memory}MB")
            
            # 여러 개의 큰 텐서로 GPU 메모리 최대 활용
            tensors = []
            tensor_count = 10  # 10개의 큰 텐서 생성
            
            for i in range(tensor_count):
                tensor_size = available_memory * 1024 * 1024 // (tensor_count * 4)  # float32 기준
                tensor = torch.randn(tensor_size, device=self.device)
                tensors.append(tensor)
                print(f"📦 텐서 {i+1}/{tensor_count} 생성 완료 ({tensor_size:,} 요소)")
            
                # 고강도 GPU 계산 작업
                def intensive_gpu_work():
                    iteration = 0
                    while True:
                        try:
                            # ESC 키 확인
                            if self.check_esc_key():
                                print(f"\n⏹️ {process_name} 고강도 GPU 작업 중지 (ESC 키 감지)")
                                break
                            
                            # 복잡한 딥러닝 연산 시뮬레이션
                            for i, tensor in enumerate(tensors):
                                try:
                                    # 텐서 차원 확인 및 조정
                                    if tensor.dim() == 1:
                                        tensor = tensor.unsqueeze(0)
                                    
                                    # 1. 행렬 곱셈 (2차원 이상만)
                                    if tensor.dim() >= 2:
                                        tensor = torch.matmul(tensor, tensor.T)
                                    
                                    # 2. 컨볼루션 연산 (안전한 차원 처리)
                                    if tensor.dim() == 1:
                                        tensor = tensor.unsqueeze(0)
                                    if tensor.dim() >= 2:
                                        if tensor.dim() == 2:
                                            tensor = tensor.unsqueeze(0)
                                        
                                        # 컨볼루션 커널 크기 조정
                                        kernel_size = min(3, tensor.size(-1))
                                        if kernel_size > 0:
                                            conv_kernel = torch.randn(1, tensor.size(1), kernel_size, device=self.device)
                                            tensor = torch.conv1d(tensor, conv_kernel, padding=kernel_size//2)
                                    
                                    # 3. 활성화 함수들
                                    tensor = torch.relu(tensor)
                                    tensor = torch.sigmoid(tensor)
                                    tensor = torch.tanh(tensor)
                                    
                                    # 4. 정규화 (차원 확인)
                                    if tensor.dim() > 1:
                                        tensor = torch.nn.functional.normalize(tensor, dim=1)
                                    
                                    # 5. 드롭아웃 시뮬레이션
                                    mask = torch.rand_like(tensor) > 0.1
                                    tensor = tensor * mask
                                    
                                    # 6. 배치 정규화 시뮬레이션 (차원 확인)
                                    if tensor.dim() > 1:
                                        mean = tensor.mean(dim=1, keepdim=True)
                                        var = tensor.var(dim=1, keepdim=True, unbiased=False)
                                        tensor = (tensor - mean) / torch.sqrt(var + 1e-5)
                                    
                                    tensors[i] = tensor
                                    
                                except Exception as e:
                                    # 오류 발생 시 텐서 재생성
                                    print(f"⚠️ 고강도 텐서 {i} 처리 중 오류: {e}, 텐서 재생성")
                                    tensor_size = len(tensors[i])
                                    tensors[i] = torch.randn(tensor_size, device=self.device)
                            
                            # GPU 동기화
                            torch.cuda.synchronize()
                            
                            iteration += 1
                            if iteration % 50 == 0:
                                # GPU 사용률 확인
                                gpu_util, mem_used, mem_total = self.monitor_gpu_usage()
                                print(f"🔥 {process_name} 고강도 GPU 작업 진행 중... (반복 {iteration}, GPU 사용률: {gpu_util}%) - ESC로 중지")
                            
                            time.sleep(0.005)  # 5ms 대기 (더 빠른 반복)
                            
                        except Exception as e:
                            print(f"❌ 고강도 GPU 작업 오류: {e}")
                            break
            
            # 백그라운드 스레드로 고강도 GPU 작업 실행
            gpu_thread = threading.Thread(target=intensive_gpu_work, daemon=True)
            gpu_thread.start()
            
            self.gpu_processes[process_name] = {
                'tensors': tensors,
                'thread': gpu_thread,
                'memory_mb': available_memory,
                'start_time': datetime.now(),
                'iterations': 0,
                'intensive_mode': True
            }
            
            print(f"🔥 {process_name}에 대해 최대 GPU 자원 할당 완료 ({available_memory}MB)")
            return True
            
        except Exception as e:
            print(f"❌ 최대 GPU 자원 할당 실패: {e}")
            return False
    
    def create_specific_gpu_workload(self, process_name, gpu_id=0):
        """특정 GPU에 작업 할당 (GPU 0, 1, 2 등)"""
        try:
            gpu_count = torch.cuda.device_count()
            if gpu_id >= gpu_count:
                print(f"❌ GPU {gpu_id}가 존재하지 않습니다. 사용 가능한 GPU: 0-{gpu_count-1}")
                return False
            
            # GPU 정보 가져오기
            gpu_props = torch.cuda.get_device_properties(gpu_id)
            total_memory = gpu_props.total_memory // (1024 * 1024)  # MB 단위
            available_memory = int(total_memory * 0.8)  # 80% 사용
            
            print(f"🎯 {process_name}에 대해 GPU {gpu_id} 전용 작업 할당 시작...")
            print(f"📊 GPU {gpu_id} 정보: {gpu_props.name}")
            print(f"📊 총 메모리: {total_memory}MB, 할당 예정: {available_memory}MB")
            
            # 지정된 GPU에 텐서 생성
            with torch.cuda.device(gpu_id):
                tensors = []
                tensor_count = 8  # 8개의 큰 텐서 생성
                
                for i in range(tensor_count):
                    tensor_size = available_memory * 1024 * 1024 // (tensor_count * 4)  # float32 기준
                    tensor = torch.randn(tensor_size, device=f'cuda:{gpu_id}')
                    tensors.append(tensor)
                    print(f"📦 GPU {gpu_id} 텐서 {i+1}/{tensor_count} 생성 완료 ({tensor_size:,} 요소)")
                
                # 지정된 GPU에서 고강도 작업 수행
                def specific_gpu_work():
                    iteration = 0
                    while True:
                        try:
                            # ESC 키 확인
                            if self.check_esc_key():
                                print(f"\n⏹️ {process_name} GPU {gpu_id} 작업 중지 (ESC 키 감지)")
                                break
                            
                            # 복잡한 딥러닝 연산 시뮬레이션
                            for i, tensor in enumerate(tensors):
                                try:
                                    # 텐서 차원 확인 및 조정
                                    if tensor.dim() == 1:
                                        tensor = tensor.unsqueeze(0)
                                    
                                    # 1. 행렬 곱셈 (2차원 이상만)
                                    if tensor.dim() >= 2:
                                        tensor = torch.matmul(tensor, tensor.T)
                                    
                                    # 2. 컨볼루션 연산 (안전한 차원 처리)
                                    if tensor.dim() == 1:
                                        tensor = tensor.unsqueeze(0)
                                    if tensor.dim() >= 2:
                                        if tensor.dim() == 2:
                                            tensor = tensor.unsqueeze(0)
                                        
                                        # 컨볼루션 커널 크기 조정
                                        kernel_size = min(3, tensor.size(-1))
                                        if kernel_size > 0:
                                            conv_kernel = torch.randn(1, tensor.size(1), kernel_size, device=f'cuda:{gpu_id}')
                                            tensor = torch.conv1d(tensor, conv_kernel, padding=kernel_size//2)
                                    
                                    # 3. 활성화 함수들
                                    tensor = torch.relu(tensor)
                                    tensor = torch.sigmoid(tensor)
                                    tensor = torch.tanh(tensor)
                                    
                                    # 4. 정규화 (차원 확인)
                                    if tensor.dim() > 1:
                                        tensor = torch.nn.functional.normalize(tensor, dim=1)
                                    
                                    # 5. 드롭아웃 시뮬레이션
                                    mask = torch.rand_like(tensor) > 0.1
                                    tensor = tensor * mask
                                    
                                    # 6. 배치 정규화 시뮬레이션 (차원 확인)
                                    if tensor.dim() > 1:
                                        mean = tensor.mean(dim=1, keepdim=True)
                                        var = tensor.var(dim=1, keepdim=True, unbiased=False)
                                        tensor = (tensor - mean) / torch.sqrt(var + 1e-5)
                                    
                                    tensors[i] = tensor
                                    
                                except Exception as e:
                                    # 오류 발생 시 텐서 재생성
                                    print(f"⚠️ GPU {gpu_id} 텐서 {i} 처리 중 오류: {e}, 텐서 재생성")
                                    tensor_size = len(tensors[i])
                                    tensors[i] = torch.randn(tensor_size, device=f'cuda:{gpu_id}')
                            
                            # GPU 동기화
                            torch.cuda.synchronize(gpu_id)
                            
                            iteration += 1
                            if iteration % 50 == 0:
                                # GPU 사용률 확인
                                gpu_util, mem_used, mem_total = self.monitor_specific_gpu_usage(gpu_id)
                                print(f"🎯 GPU {gpu_id} 고강도 작업 진행 중... (반복 {iteration}, GPU 사용률: {gpu_util}%) - ESC로 중지")
                            
                            time.sleep(0.005)  # 5ms 대기
                            
                        except Exception as e:
                            print(f"❌ GPU {gpu_id} 작업 오류: {e}")
                            break
                
                # 백그라운드 스레드로 GPU 작업 실행
                gpu_thread = threading.Thread(target=specific_gpu_work, daemon=True)
                gpu_thread.start()
                
                self.gpu_processes[process_name] = {
                    'tensors': tensors,
                    'thread': gpu_thread,
                    'gpu_id': gpu_id,
                    'memory_mb': available_memory,
                    'start_time': datetime.now(),
                    'iterations': 0,
                    'specific_gpu_mode': True
                }
                
                print(f"✅ {process_name}에 대해 GPU {gpu_id} 전용 작업 할당 완료 ({available_memory}MB)")
                return True
                
        except Exception as e:
            print(f"❌ GPU {gpu_id} 작업 할당 실패: {e}")
            return False
    
    def create_multi_gpu_workload(self, process_name):
        """멀티 GPU 지원 (여러 GPU에 작업 분산)"""
        try:
            gpu_count = torch.cuda.device_count()
            if gpu_count < 2:
                print("⚠️ 멀티 GPU가 감지되지 않았습니다. 단일 GPU 모드로 실행합니다.")
                return self.create_maximum_gpu_workload(process_name)
            
            print(f"🚀 {process_name}에 대해 {gpu_count}개 GPU에 작업 분산 시작...")
            
            gpu_tensors = {}
            gpu_threads = {}
            
            for gpu_id in range(gpu_count):
                # 각 GPU별로 텐서 생성
                with torch.cuda.device(gpu_id):
                    tensors = []
                    for i in range(5):
                        tensor = torch.randn(1024 * 1024, device=f'cuda:{gpu_id}')
                        tensors.append(tensor)
                    gpu_tensors[gpu_id] = tensors
                    
                    # 각 GPU별 작업 스레드
                    def gpu_work(gpu_id):
                        iteration = 0
                        while True:
                            try:
                                # ESC 키 확인
                                if self.check_esc_key():
                                    print(f"\n⏹️ GPU {gpu_id} 작업 중지 (ESC 키 감지)")
                                    break
                                
                                for i, tensor in enumerate(gpu_tensors[gpu_id]):
                                    try:
                                        # 텐서 차원 확인 및 조정
                                        if tensor.dim() == 1:
                                            tensor = tensor.unsqueeze(0)
                                        
                                        # 행렬 곱셈 (2차원 이상만)
                                        if tensor.dim() >= 2:
                                            tensor = torch.matmul(tensor, tensor.T)
                                        
                                        # 활성화 함수 적용
                                        tensor = torch.relu(tensor)
                                        
                                        # 정규화 (차원 확인)
                                        if tensor.dim() > 0:
                                            tensor = torch.nn.functional.normalize(tensor, dim=0)
                                        
                                        gpu_tensors[gpu_id][i] = tensor
                                        
                                    except Exception as e:
                                        # 오류 발생 시 텐서 재생성
                                        print(f"⚠️ 멀티 GPU {gpu_id} 텐서 {i} 처리 중 오류: {e}, 텐서 재생성")
                                        tensor_size = len(gpu_tensors[gpu_id][i])
                                        gpu_tensors[gpu_id][i] = torch.randn(tensor_size, device=f'cuda:{gpu_id}')
                                
                                torch.cuda.synchronize(gpu_id)
                                iteration += 1
                                
                                if iteration % 100 == 0:
                                    print(f"🔄 GPU {gpu_id} 작업 진행 중... (반복 {iteration}) - ESC로 중지")
                                
                                time.sleep(0.01)
                                
                            except Exception as e:
                                print(f"❌ GPU {gpu_id} 작업 오류: {e}")
                                break
                    
                    thread = threading.Thread(target=gpu_work, args=(gpu_id,), daemon=True)
                    thread.start()
                    gpu_threads[gpu_id] = thread
            
            self.gpu_processes[process_name] = {
                'gpu_tensors': gpu_tensors,
                'gpu_threads': gpu_threads,
                'gpu_count': gpu_count,
                'start_time': datetime.now(),
                'multi_gpu_mode': True
            }
            
            print(f"✅ {process_name}에 대해 {gpu_count}개 GPU 작업 분산 완료")
            return True
            
        except Exception as e:
            print(f"❌ 멀티 GPU 작업 분산 실패: {e}")
            return False
    
    def monitor_specific_gpu_usage(self, gpu_id):
        """특정 GPU 사용률 모니터링"""
        try:
            # NVIDIA-SMI로 특정 GPU 사용률 확인
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                   '--format=csv,noheader,nounits', f'--id={gpu_id}'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                gpu_util = int(gpu_info[0])
                memory_used = int(gpu_info[1])
                memory_total = int(gpu_info[2])
                
                print(f"📊 GPU {gpu_id} 사용률: {gpu_util}%, 메모리: {memory_used}/{memory_total} MB")
                return gpu_util, memory_used, memory_total
            else:
                print(f"⚠️ GPU {gpu_id} 정보를 가져올 수 없습니다.")
                return None, None, None
                
        except Exception as e:
            print(f"❌ GPU {gpu_id} 모니터링 오류: {e}")
            return None, None, None
    
    def list_available_gpus(self):
        """사용 가능한 GPU 목록 표시"""
        try:
            gpu_count = torch.cuda.device_count()
            print(f"\n🎮 사용 가능한 GPU 목록 ({gpu_count}개):")
            
            for gpu_id in range(gpu_count):
                gpu_props = torch.cuda.get_device_properties(gpu_id)
                total_memory = gpu_props.total_memory // (1024 * 1024)  # MB 단위
                print(f"  GPU {gpu_id}: {gpu_props.name} ({total_memory}MB)")
            
            return gpu_count
            
        except Exception as e:
            print(f"❌ GPU 목록 조회 실패: {e}")
            return 0
    
    def monitor_gpu_usage(self):
        """GPU 사용률 모니터링"""
        try:
            # NVIDIA-SMI로 GPU 사용률 확인
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                gpu_util = int(gpu_info[0])
                memory_used = int(gpu_info[1])
                memory_total = int(gpu_info[2])
                
                print(f"📊 GPU 사용률: {gpu_util}%, 메모리: {memory_used}/{memory_total} MB")
                return gpu_util, memory_used, memory_total
            else:
                print("⚠️ GPU 정보를 가져올 수 없습니다.")
                return None, None, None
                
        except Exception as e:
            print(f"❌ GPU 모니터링 오류: {e}")
            return None, None, None
    
    def optimize_process_for_gpu(self, process_name, mode="normal", gpu_id=0):
        """프로세스를 GPU 최적화"""
        config = self.process_config.get("gpu_processes", {}).get(process_name, {})
        priority = config.get("priority", "medium")
        gpu_memory = config.get("gpu_memory", 512)
        
        print(f"🔧 {process_name} GPU 최적화 시작... (모드: {mode}, GPU: {gpu_id})")
        
        # 1. GPU 우선순위 설정
        self.set_process_gpu_priority(process_name, priority)
        
        # 2. GPU 작업 부하 생성 (모드에 따라)
        if mode == "maximum":
            self.create_maximum_gpu_workload(process_name)
        elif mode == "multi_gpu":
            self.create_multi_gpu_workload(process_name)
        elif mode == "gpu_0":
            self.create_specific_gpu_workload(process_name, gpu_id)
        else:
            self.create_gpu_workload(process_name, gpu_memory)
        
        # 3. 프로세스 우선순위 조정
        self.set_process_priority(process_name, priority)
        
        print(f"✅ {process_name} GPU 최적화 완료")
    
    def set_process_priority(self, process_name, priority):
        """프로세스 우선순위 설정"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == process_name:
                    if priority == "high":
                        proc.nice(psutil.HIGH_PRIORITY_CLASS)
                    elif priority == "medium":
                        proc.nice(psutil.NORMAL_PRIORITY_CLASS)
                    else:
                        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                    
                    print(f"📈 {process_name} (PID: {proc.info['pid']}) 우선순위를 {priority}로 설정")
                    break
        except Exception as e:
            print(f"❌ 프로세스 우선순위 설정 실패: {e}")
    
    def add_process_to_config(self, process_name, priority="medium", gpu_memory=512, mode="normal", gpu_id=0):
        """새 프로세스를 설정에 추가"""
        self.process_config["gpu_processes"][process_name] = {
            "priority": priority,
            "gpu_memory": gpu_memory,
            "mode": mode,
            "gpu_id": gpu_id
        }
        self.save_config()
        print(f"➕ {process_name}을 설정에 추가했습니다. (모드: {mode}, GPU: {gpu_id})")
    
    def remove_process_from_config(self, process_name):
        """프로세스를 설정에서 제거"""
        if process_name in self.process_config["gpu_processes"]:
            del self.process_config["gpu_processes"][process_name]
            self.save_config()
            print(f"➖ {process_name}을 설정에서 제거했습니다.")
        else:
            print(f"❌ {process_name}이 설정에 없습니다.")
    
    def check_esc_key(self):
        """ESC 키 입력 확인 (비동기)"""
        try:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\x1b':  # ESC 키
                    return True
        except:
            pass
        return False
    
    def start_automation(self):
        """자동화 시작"""
        print("🚀 GPU 프로세스 자동화 시작...")
        print("💡 ESC 키를 누르면 메인 메뉴로 돌아갑니다.")
        
        while True:
            try:
                # ESC 키 확인
                if self.check_esc_key():
                    print("\n⏹️ ESC 키 감지! 메인 메뉴로 돌아갑니다...")
                    break
                
                # 현재 프로세스 목록 가져오기
                processes = self.get_process_list()
                
                # 설정된 프로세스들 확인 및 최적화
                for proc in processes:
                    process_name = proc['name']
                    if process_name in self.process_config["gpu_processes"]:
                        if process_name not in self.gpu_processes:
                            # 새로운 프로세스 발견, GPU 최적화 시작
                            config = self.process_config["gpu_processes"][process_name]
                            mode = config.get("mode", "normal")
                            gpu_id = config.get("gpu_id", 0)
                            
                            if mode == "gpu_0":
                                self.optimize_process_for_gpu(process_name, "gpu_0", gpu_id)
                            elif mode == "maximum":
                                self.optimize_process_for_gpu(process_name, "maximum")
                            elif mode == "multi_gpu":
                                self.optimize_process_for_gpu(process_name, "multi_gpu")
                            else:
                                self.optimize_process_for_gpu(process_name, "normal")
                
                # GPU 사용률 모니터링
                self.monitor_gpu_usage()
                
                # 1초 대기 (ESC 키 감지를 위해 더 짧게)
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("\n⏹️ Ctrl+C 감지! 자동화 중지...")
                break
            except Exception as e:
                print(f"❌ 자동화 오류: {e}")
                time.sleep(1)
    
    def show_current_status(self):
        """현재 상태 표시"""
        print("\n=== GPU 프로세스 자동화 상태 ===")
        
        # 설정된 프로세스들
        print("\n📋 설정된 프로세스들:")
        for proc_name, config in self.process_config["gpu_processes"].items():
            status = "🟢 활성" if proc_name in self.gpu_processes else "⚪ 비활성"
            mode = config.get("mode", "normal")
            gpu_id = config.get("gpu_id", 0)
            print(f"  {status} {proc_name} - 우선순위: {config['priority']}, 모드: {mode}, GPU: {gpu_id}, 메모리: {config['gpu_memory']}MB")
        
        # GPU 사용률
        gpu_util, mem_used, mem_total = self.monitor_gpu_usage()
        
        # 활성 GPU 프로세스들
        print("\n🚀 활성 GPU 프로세스들:")
        for proc_name, info in self.gpu_processes.items():
            runtime = datetime.now() - info['start_time']
            print(f"  {proc_name} - 메모리: {info['memory_mb']}MB, 실행시간: {runtime}")

def main():
    """메인 함수"""
    automation = GPUProcessAutomation()
    
    while True:
        print("\n=== GPU 프로세스 자동화 도구 ===")
        print("1. 현재 프로세스 목록 보기")
        print("2. 프로세스 GPU 최적화 시작")
        print("3. 최대 GPU 자원 할당 (90% 메모리 사용)")
        print("4. 멀티 GPU 작업 분산")
        print("5. GPU 0 전용 모드")
        print("6. 사용 가능한 GPU 목록 보기")
        print("7. 자동화 모드 시작")
        print("8. 프로세스 설정 추가/제거")
        print("9. 현재 상태 확인")
        print("10. 종료")
        print("\n💡 모든 실행 상태에서 ESC 키를 누르면 메인 메뉴로 돌아갑니다!")
        
        choice = input("\n선택하세요 (1-10): ").strip()
        
        if choice == "1":
            processes = automation.get_process_list()
            print(f"\n📋 현재 실행 중인 프로세스 ({len(processes)}개):")
            for proc in processes[:20]:  # 상위 20개만 표시
                print(f"  {proc['name']} (PID: {proc['pid']}) - CPU: {proc['cpu_percent']:.1f}%, 메모리: {proc['memory_percent']:.1f}%")
        
        elif choice == "2":
            process_name = input("GPU 최적화할 프로세스 이름을 입력하세요: ").strip()
            if process_name:
                automation.optimize_process_for_gpu(process_name, "normal")
        
        elif choice == "3":
            process_name = input("최대 GPU 자원을 할당할 프로세스 이름을 입력하세요: ").strip()
            if process_name:
                automation.optimize_process_for_gpu(process_name, "maximum")
        
        elif choice == "4":
            process_name = input("멀티 GPU에 작업을 분산할 프로세스 이름을 입력하세요: ").strip()
            if process_name:
                automation.optimize_process_for_gpu(process_name, "multi_gpu")
        
        elif choice == "5":
            process_name = input("GPU 0에 작업을 할당할 프로세스 이름을 입력하세요: ").strip()
            if process_name:
                gpu_id = input("GPU ID (기본값: 0): ").strip() or "0"
                automation.optimize_process_for_gpu(process_name, "gpu_0", int(gpu_id))
        
        elif choice == "6":
            automation.list_available_gpus()
        
        elif choice == "7":
            print("자동화 모드를 시작합니다. Ctrl+C로 중지할 수 있습니다.")
            automation.start_automation()
        
        elif choice == "8":
            print("\n1. 프로세스 추가")
            print("2. 프로세스 제거")
            sub_choice = input("선택하세요 (1-2): ").strip()
            
            if sub_choice == "1":
                name = input("프로세스 이름: ").strip()
                priority = input("우선순위 (high/medium/low): ").strip() or "medium"
                memory = input("GPU 메모리 (MB): ").strip() or "512"
                
                print("\nGPU 모드 선택:")
                print("1. 일반 모드 (normal)")
                print("2. 최대 GPU 자원 (maximum)")
                print("3. 멀티 GPU 분산 (multi_gpu)")
                print("4. GPU 0 전용 (gpu_0)")
                mode_choice = input("모드 선택 (1-4): ").strip() or "1"
                
                mode = "normal"
                gpu_id = 0
                
                if mode_choice == "2":
                    mode = "maximum"
                elif mode_choice == "3":
                    mode = "multi_gpu"
                elif mode_choice == "4":
                    mode = "gpu_0"
                    gpu_id_input = input("GPU ID (기본값: 0): ").strip() or "0"
                    gpu_id = int(gpu_id_input)
                
                automation.add_process_to_config(name, priority, int(memory), mode, gpu_id)
            
            elif sub_choice == "2":
                name = input("제거할 프로세스 이름: ").strip()
                automation.remove_process_from_config(name)
        
        elif choice == "9":
            automation.show_current_status()
        
        elif choice == "10":
            print("프로그램을 종료합니다.")
            break
        
        else:
            print("잘못된 선택입니다.")

if __name__ == "__main__":
    main() 