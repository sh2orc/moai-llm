#!/usr/bin/env python3
"""
토크나이저 속도 벤치마크 v2
다양한 조건에서 테스트하여 병목 지점 파악
"""
import os
import time
import multiprocessing
from transformers import AutoTokenizer

def test_tokenization(parallelism_setting, text_length, num_samples=10000):
    """특정 조건에서 토크나이징 속도 테스트"""
    # TOKENIZERS_PARALLELISM 설정
    os.environ["TOKENIZERS_PARALLELISM"] = parallelism_setting
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B",
        use_fast=True,
    )
    
    # 테스트 데이터 생성
    base_text = "This is a sample text for tokenization. "
    test_texts = [base_text * text_length] * num_samples
    
    # 토크나이징
    start = time.time()
    result = tokenizer(test_texts, truncation=False, padding=False)
    elapsed = time.time() - start
    speed = num_samples / elapsed
    
    # 토큰 통계
    total_tokens = sum(len(ids) for ids in result['input_ids'])
    avg_tokens = total_tokens / num_samples
    
    return {
        'speed': speed,
        'elapsed': elapsed,
        'avg_tokens': avg_tokens,
        'tokens_per_sec': total_tokens / elapsed
    }

print("="*80)
print("🧪 Fast Tokenizer 속도 진단 테스트")
print("="*80)
print()

# 시스템 정보
cpu_count = multiprocessing.cpu_count()
print(f"💻 CPU Cores: {cpu_count}")
print(f"🔧 Python: {os.sys.version.split()[0]}")
print()

# 토크나이저 정보
os.environ["TOKENIZERS_PARALLELISM"] = "true"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", use_fast=True)
print(f"✓ Fast Tokenizer: {tokenizer.is_fast}")
print(f"✓ Type: {type(tokenizer).__name__}")
print()

print("="*80)
print("📊 TEST 1: TOKENIZERS_PARALLELISM 영향")
print("="*80)

for setting in ["false", "true"]:
    result = test_tokenization(setting, text_length=10, num_samples=10000)
    print(f"\nTOKENIZERS_PARALLELISM={setting}")
    print(f"  Speed: {result['speed']:,.0f} examples/s")
    print(f"  Tokens/sec: {result['tokens_per_sec']:,.0f}")
    print(f"  Avg tokens: {result['avg_tokens']:.0f}")

print()
print("="*80)
print("📊 TEST 2: 텍스트 길이 영향 (PARALLELISM=true)")
print("="*80)

for length in [1, 5, 10, 20, 50]:
    result = test_tokenization("true", text_length=length, num_samples=10000)
    print(f"\nText length: {length}x (≈{length * 40} chars)")
    print(f"  Speed: {result['speed']:,.0f} examples/s")
    print(f"  Tokens/sec: {result['tokens_per_sec']:,.0f}")
    print(f"  Avg tokens: {result['avg_tokens']:.0f}")

print()
print("="*80)
print("📊 TEST 3: 샘플 수 영향 (PARALLELISM=true, length=10x)")
print("="*80)

for num_samples in [1000, 5000, 10000, 20000]:
    result = test_tokenization("true", text_length=10, num_samples=num_samples)
    print(f"\nSamples: {num_samples:,}")
    print(f"  Speed: {result['speed']:,.0f} examples/s")
    print(f"  Tokens/sec: {result['tokens_per_sec']:,.0f}")
    print(f"  Time: {result['elapsed']:.2f}s")

print()
print("="*80)
print("📊 TEST 4: 짧은 텍스트로 최대 성능 측정")
print("="*80)

# 실제 학습 데이터와 유사한 짧은 텍스트
result = test_tokenization("true", text_length=1, num_samples=50000)
print(f"\n50,000 samples, 매우 짧은 텍스트 (≈40 chars)")
print(f"  Speed: {result['speed']:,.0f} examples/s")
print(f"  Tokens/sec: {result['tokens_per_sec']:,.0f}")
print(f"  Time: {result['elapsed']:.2f}s")

print()
print("="*80)
print("📝 결론:")
print("="*80)

if result['speed'] > 30000:
    print("✅ Fast Tokenizer가 정상 작동 (30,000+ examples/s)")
    print("   → datasets.map() 오버헤드가 문제")
    print("   → Multiprocessing이 필요")
elif result['speed'] > 10000:
    print("⚠️  Fast Tokenizer가 부분적으로 작동 (10,000-30,000 examples/s)")
    print("   → TOKENIZERS_PARALLELISM 설정 확인 필요")
    print("   → 텍스트 길이나 배치 크기 조정 필요")
else:
    print("❌ Fast Tokenizer가 기대만큼 빠르지 않음 (<10,000 examples/s)")
    print("   → 시스템 리소스 제한 가능성")
    print("   → Multiprocessing이 반드시 필요")

print()
print("💡 권장사항:")
if result['speed'] < 15000:
    print("   - num_proc=8로 multiprocessing 활성화")
    print("   - batch_size=50000으로 설정")
    print("   - 예상 속도: 8 × 6,000 = 48,000 examples/s")
    print("   - 750만 샘플: 약 2.5분")
else:
    print("   - num_proc=4로 적당한 multiprocessing")
    print("   - batch_size=100000으로 설정")
    print("   - 예상 속도: 4 × 15,000 = 60,000 examples/s")
    print("   - 750만 샘플: 약 2분")

print("="*80)

