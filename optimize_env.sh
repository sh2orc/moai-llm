#!/bin/bash
# 토크나이징 최적화 환경 변수 설정

# 고성능 CPU 환경 (32+ cores)
export TOKENIZERS_PARALLELISM=true  # Rust 토크나이저 병렬화 활성화
export RAYON_NUM_THREADS=48  # Rust 병렬 처리 스레드 수
export OMP_NUM_THREADS=48  # OpenMP 스레드 수

# Dataset 처리 최적화
export DATASET_NUM_PROC=48  # 병렬 프로세스 수
export DATASET_BATCH_SIZE=20000  # 배치 크기
export DATASET_WRITER_BATCH_SIZE=100000  # Writer 버퍼 크기

# PyArrow 최적화
export ARROW_DEFAULT_MEMORY_POOL=mimalloc  # 더 빠른 메모리 할당자
export ARROW_IO_THREADS=16  # I/O 스레드 수

echo "✅ 최적화 환경 변수 설정 완료"
echo "📊 병렬 처리: 48 프로세스"
echo "📦 배치 크기: 20000"
echo "💾 Writer 버퍼: 100000"

