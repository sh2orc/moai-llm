#!/usr/bin/env python3
"""
토크나이저 속도 벤치마크 테스트
GIL 우회가 제대로 작동하는지 확인
"""
import os
import time
from transformers import AutoTokenizer

# TOKENIZERS_PARALLELISM 활성화
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 토크나이저 로드
print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B",
    use_fast=True,
)

# Fast Tokenizer 확인
print(f"✓ Is Fast Tokenizer: {tokenizer.is_fast}")
print(f"✓ Tokenizer type: {type(tokenizer)}")
print()

# 테스트 데이터 생성 (실제 데이터와 유사)
test_texts = [
    "This is a sample text for tokenization. " * 50  # ~50 words
] * 10000  # 10,000 samples

print(f"📊 Testing with {len(test_texts):,} samples...")
print()

# 테스트 1: 단순 토크나이징 (Python GIL 영향 최소)
print("Test 1: Pure tokenization (minimal Python overhead)")
start = time.time()
result = tokenizer(test_texts, truncation=False, padding=False)
elapsed = time.time() - start
speed = len(test_texts) / elapsed
print(f"  Time: {elapsed:.2f}s")
print(f"  Speed: {speed:,.0f} examples/s")
print()

# 테스트 2: 배치 토크나이징 (datasets.map과 유사)
print("Test 2: Batched processing (similar to datasets.map)")
batch_size = 10000
num_batches = len(test_texts) // batch_size

start = time.time()
for i in range(num_batches):
    batch = test_texts[i * batch_size:(i + 1) * batch_size]
    _ = tokenizer(batch, truncation=False, padding=False)
elapsed = time.time() - start
speed = len(test_texts) / elapsed
print(f"  Time: {elapsed:.2f}s")
print(f"  Speed: {speed:,.0f} examples/s")
print()

# 테스트 3: 작은 배치 (datasets.map 기본 동작)
print("Test 3: Small batches (1000 samples per batch)")
batch_size = 1000
num_batches = len(test_texts) // batch_size

start = time.time()
for i in range(num_batches):
    batch = test_texts[i * batch_size:(i + 1) * batch_size]
    _ = tokenizer(batch, truncation=False, padding=False)
elapsed = time.time() - start
speed = len(test_texts) / elapsed
print(f"  Time: {elapsed:.2f}s")
print(f"  Speed: {speed:,.0f} examples/s")
print()

print("="*60)
print("📝 결론:")
print("  - Fast Tokenizer가 50,000+ examples/s를 달성한다면:")
print("    → GIL 문제 없음, 단일 프로세스로도 충분")
print("  - Fast Tokenizer가 여전히 느리다면:")
print("    → datasets 라이브러리 오버헤드가 병목")
print("    → Multiprocessing 병행이 필요")
print("="*60)

