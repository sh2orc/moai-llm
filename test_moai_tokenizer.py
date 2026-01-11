#!/usr/bin/env python3
"""MOAI 토크나이저 테스트 스크립트"""

from transformers import AutoTokenizer

def main():
    print("=" * 60)
    print("🧪 MOAI 토크나이저 테스트")
    print("=" * 60)
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained('./tokenizers/moai')
    
    print(f"\n📊 토크나이저 정보:")
    print(f"   - Vocab 크기: {tokenizer.vocab_size:,}")
    print(f"   - PAD: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"   - BOS: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"   - EOS: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"   - UNK: {tokenizer.unk_token} (id={tokenizer.unk_token_id})")
    
    # 테스트 케이스
    test_cases = [
        ("한국어 기본", "안녕하세요. 토크나이저 테스트입니다."),
        ("영어 기본", "Hello, world! This is a test."),
        ("BC카드 금융", "BC카드 신용카드 결제 서비스 안내"),
        ("금융 용어", "신용대출 금리 및 할부 서비스 문의"),
        ("혼합 텍스트", "Hello 안녕 金融 finance 테스트"),
        ("코드", "def hello(): print('Hello, World!')"),
        ("숫자", "2024년 1월 15일 금액: 1,234,567원"),
    ]
    
    print("\n" + "=" * 60)
    print("📝 토큰화 테스트")
    print("=" * 60)
    
    for name, text in test_cases:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text, add_special_tokens=False)
        
        print(f"\n[{name}]")
        print(f"   입력: {text}")
        print(f"   토큰 수: {len(tokens)}")
        print(f"   토큰: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"   ID: {ids[:10]}{'...' if len(ids) > 10 else ''}")
    
    # 인코드/디코드 테스트
    print("\n" + "=" * 60)
    print("🔄 인코드/디코드 왕복 테스트")
    print("=" * 60)
    
    test_text = "BC카드 금융 서비스: Hello World! 신용카드 결제"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\n   원본: {test_text}")
    print(f"   인코딩: {encoded}")
    print(f"   디코딩: {decoded}")
    print(f"   일치: {'✅' if test_text == decoded else '❌'}")
    
    # 채팅 템플릿 테스트
    print("\n" + "=" * 60)
    print("💬 특수 토큰 테스트")
    print("=" * 60)
    
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"   {token}: id={token_id}")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
