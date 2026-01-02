#!/usr/bin/env python3
"""
토크나이징 최적화 검증 스크립트

사용법:
    python verify_optimization.py --tokenizer_path tokenizers/moai

이 스크립트는 다음을 확인합니다:
1. Fast Tokenizer 활성화 여부
2. CPU 코어 수
3. 사용 가능한 RAM
4. 디스크 타입 (SSD vs HDD)
5. 최적화 환경 변수 설정 상태
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️ psutil이 설치되지 않았습니다. 메모리 정보를 확인할 수 없습니다.")
    print("   설치: pip install psutil")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("❌ transformers가 설치되지 않았습니다.")
    sys.exit(1)


def check_tokenizer(tokenizer_path: str):
    """토크나이저 Fast 모드 확인"""
    print("\n" + "="*80)
    print("1. Fast Tokenizer 확인")
    print("="*80)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        if tokenizer.is_fast:
            print("✅ Fast Tokenizer (Rust 기반) 활성화됨")
            print(f"   경로: {tokenizer_path}")
            return True
        else:
            print("⚠️ Slow Tokenizer (Python 기반) 사용 중")
            print("   → 토크나이징이 10-50배 느립니다!")
            print("   → Fast Tokenizer 지원 토크나이저로 교체를 권장합니다.")
            return False
    except Exception as e:
        print(f"❌ 토크나이저 로드 실패: {e}")
        return False


def check_cpu():
    """CPU 코어 수 확인"""
    print("\n" + "="*80)
    print("2. CPU 코어 수 확인")
    print("="*80)
    
    try:
        cpu_count = os.cpu_count()
        print(f"✅ CPU 코어 수: {cpu_count}")
        
        if cpu_count >= 48:
            print("   → 고성능 CPU: num_proc=48 권장")
        elif cpu_count >= 32:
            print("   → 중급 CPU: num_proc=32-48 권장")
        elif cpu_count >= 16:
            print("   → 일반 CPU: num_proc=16-32 권장")
        else:
            print("   → 저성능 CPU: num_proc=8-16 권장")
        
        return cpu_count
    except Exception as e:
        print(f"❌ CPU 정보 확인 실패: {e}")
        return None


def check_memory():
    """사용 가능한 RAM 확인"""
    print("\n" + "="*80)
    print("3. 메모리 (RAM) 확인")
    print("="*80)
    
    if not HAS_PSUTIL:
        print("⚠️ psutil 미설치로 메모리 정보를 확인할 수 없습니다.")
        return None
    
    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        print(f"✅ 전체 메모리: {total_gb:.1f} GB")
        print(f"   사용 가능: {available_gb:.1f} GB ({mem.percent:.1f}% 사용 중)")
        
        if available_gb > 50:
            print("   → 충분한 RAM: 스마트 메모리 활용 모드 사용 가능")
        elif available_gb > 20:
            print("   → 보통 RAM: 기본 설정 사용")
        else:
            print("   → 제한된 RAM: 메모리 절약 모드 권장")
            print("      (num_proc 낮추기, batch_size 줄이기)")
        
        return available_gb
    except Exception as e:
        print(f"❌ 메모리 정보 확인 실패: {e}")
        return None


def check_disk():
    """디스크 타입 확인 (SSD vs HDD)"""
    print("\n" + "="*80)
    print("4. 디스크 타입 확인")
    print("="*80)
    
    if not HAS_PSUTIL:
        print("⚠️ psutil 미설치로 디스크 정보를 확인할 수 없습니다.")
        return None
    
    try:
        # macOS에서는 diskutil 사용
        if sys.platform == "darwin":
            result = subprocess.run(
                ["diskutil", "info", "/"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if "Solid State: Yes" in result.stdout:
                print("✅ SSD 감지됨")
                print("   → 최적 I/O 성능: writer_batch_size=100000 사용 가능")
                return "SSD"
            else:
                print("⚠️ HDD 감지됨")
                print("   → I/O 제한: writer_batch_size=50000 권장")
                return "HDD"
        else:
            # Linux에서는 /sys/block 확인
            disk_info = subprocess.run(
                ["lsblk", "-d", "-o", "name,rota"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if "0" in disk_info.stdout:  # ROTA=0 means SSD
                print("✅ SSD 감지됨")
                print("   → 최적 I/O 성능: writer_batch_size=100000 사용 가능")
                return "SSD"
            else:
                print("⚠️ HDD 감지됨")
                print("   → I/O 제한: writer_batch_size=50000 권장")
                return "HDD"
    except Exception as e:
        print(f"⚠️ 디스크 타입 자동 감지 실패: {e}")
        print("   수동으로 확인하세요 (SSD 권장)")
        return None


def check_env_vars():
    """최적화 환경 변수 확인"""
    print("\n" + "="*80)
    print("5. 환경 변수 확인")
    print("="*80)
    
    env_vars = {
        "TOKENIZERS_PARALLELISM": ("Rust 토크나이저 병렬화", "true"),
        "RAYON_NUM_THREADS": ("Rust 병렬 처리 스레드", "48"),
        "OMP_NUM_THREADS": ("OpenMP 스레드", "48"),
        "DATASET_NUM_PROC": ("Dataset 병렬 프로세스", "48"),
        "DATASET_BATCH_SIZE": ("Dataset 배치 크기", "20000"),
        "DATASET_WRITER_BATCH_SIZE": ("Writer 버퍼 크기", "100000"),
        "ARROW_DEFAULT_MEMORY_POOL": ("PyArrow 메모리 할당자", "mimalloc"),
        "ARROW_IO_THREADS": ("PyArrow I/O 스레드", "16"),
    }
    
    all_set = True
    for var, (description, recommended) in env_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {var}={value}")
            print(f"   ({description})")
        else:
            print(f"⚠️ {var} 미설정")
            print(f"   권장값: {recommended} ({description})")
            all_set = False
    
    if not all_set:
        print("\n💡 최적화 환경 변수 설정 방법:")
        print("   source optimize_env.sh")
    
    return all_set


def print_summary(results):
    """최종 요약"""
    print("\n" + "="*80)
    print("📊 최적화 상태 요약")
    print("="*80)
    
    fast_tokenizer, cpu_count, available_gb, disk_type, env_set = results
    
    score = 0
    max_score = 5
    
    # Fast Tokenizer
    if fast_tokenizer:
        print("✅ Fast Tokenizer: 활성화 (가장 중요!)")
        score += 1
    else:
        print("❌ Fast Tokenizer: 미활성화 (치명적 병목!)")
    
    # CPU
    if cpu_count and cpu_count >= 32:
        print(f"✅ CPU: {cpu_count} 코어 (충분)")
        score += 1
    elif cpu_count:
        print(f"⚠️ CPU: {cpu_count} 코어 (제한적)")
        score += 0.5
    else:
        print("❌ CPU: 확인 실패")
    
    # 메모리
    if available_gb and available_gb > 50:
        print(f"✅ 메모리: {available_gb:.1f} GB (충분)")
        score += 1
    elif available_gb and available_gb > 20:
        print(f"⚠️ 메모리: {available_gb:.1f} GB (보통)")
        score += 0.5
    else:
        print(f"❌ 메모리: {'확인 실패' if not available_gb else f'{available_gb:.1f} GB (부족)'}")
    
    # 디스크
    if disk_type == "SSD":
        print("✅ 디스크: SSD (최적)")
        score += 1
    elif disk_type == "HDD":
        print("⚠️ 디스크: HDD (느림)")
        score += 0.5
    else:
        print("⚠️ 디스크: 확인 실패")
    
    # 환경 변수
    if env_set:
        print("✅ 환경 변수: 모두 설정됨")
        score += 1
    else:
        print("⚠️ 환경 변수: 일부 미설정")
    
    print(f"\n🎯 최적화 점수: {score}/{max_score}")
    
    if score >= 4.5:
        print("✅ 훌륭합니다! 최적 성능이 예상됩니다.")
        print("   예상 토크나이징 시간: 5-10분 (750만 샘플 기준)")
    elif score >= 3:
        print("⚠️ 양호합니다. 일부 개선 여지가 있습니다.")
        print("   예상 토크나이징 시간: 10-20분 (750만 샘플 기준)")
    else:
        print("❌ 최적화가 필요합니다.")
        print("   예상 토크나이징 시간: 20분+ (750만 샘플 기준)")
    
    print("\n💡 추가 권장사항:")
    if not fast_tokenizer:
        print("   1. Fast Tokenizer 활성화 (가장 중요!)")
    if not env_set:
        print("   2. 환경 변수 설정: source optimize_env.sh")
    if disk_type == "HDD" or disk_type is None:
        print("   3. SSD 사용 권장 (I/O 성능 향상)")
    if available_gb and available_gb < 20:
        print("   4. 메모리 절약 모드: --num_proc 32, batch_size 10000")


def main():
    parser = argparse.ArgumentParser(description="토크나이징 최적화 검증")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizers/moai",
        help="토크나이저 경로 (기본값: tokenizers/moai)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 토크나이징 최적화 검증 시작")
    print("="*80)
    
    # 각 항목 확인
    fast_tokenizer = check_tokenizer(args.tokenizer_path)
    cpu_count = check_cpu()
    available_gb = check_memory()
    disk_type = check_disk()
    env_set = check_env_vars()
    
    # 최종 요약
    results = (fast_tokenizer, cpu_count, available_gb, disk_type, env_set)
    print_summary(results)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

