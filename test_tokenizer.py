#!/usr/bin/env python3
"""
MOAI 토크나이저 테스트 스크립트

사용법:
    python test_tokenizer.py
    python test_tokenizer.py --tokenizer_path tokenizers/moai
    python test_tokenizer.py --compare  # 모든 토크나이저 비교
"""

import argparse
from pathlib import Path

try:
    import orjson as json  # Rust-based, faster
except ImportError:
    import json


def load_tokenizer_info(tokenizer_path: str) -> dict:
    """토크나이저 정보 로드 (라이브러리 없이)"""
    path = Path(tokenizer_path)
    
    # tokenizer.json 찾기
    if path.is_file() and path.suffix == '.json':
        json_path = path
    elif path.is_dir():
        json_path = path / "tokenizer.json"
    else:
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    with open(json_path, 'rb') as f:  # Binary for orjson
        data = json.loads(f.read())
    
    vocab = data.get('model', {}).get('vocab', {})
    merges = data.get('model', {}).get('merges', [])
    
    return {
        'path': str(path),
        'vocab_size': len(vocab),
        'merges': len(merges),
        'vocab': vocab,
        'model_type': data.get('model', {}).get('type', 'unknown'),
    }


def test_tokenizer(tokenizer_path: str = "tokenizers/moai"):
    """토크나이저 테스트"""
    try:
        from tokenizers import Tokenizer
        has_tokenizers = True
    except ImportError:
        has_tokenizers = False
    
    info = load_tokenizer_info(tokenizer_path)
    
    print("=" * 70)
    print("🧪 MOAI Tokenizer Test")
    print("=" * 70)
    print(f"📁 Path: {info['path']}")
    print(f"📊 Vocab size: {info['vocab_size']:,}")
    print(f"🔗 Merges: {info['merges']:,}")
    print(f"🏷️  Model type: {info['model_type']}")
    print()
    
    # Special tokens 확인
    special_tokens = ['<pad>', '<s>', '</s>', '<unk>', '<|endoftext|>', '<|im_start|>', '<|im_end|>']
    print("🔤 Special Tokens:")
    for token in special_tokens:
        token_id = info['vocab'].get(token)
        status = f"id={token_id}" if token_id is not None else "❌ missing"
        print(f"   {token}: {status}")
    print()
    
    if has_tokenizers:
        # 실제 토큰화 테스트
        path = Path(tokenizer_path)
        if path.is_dir():
            json_path = path / "tokenizer.json"
        else:
            json_path = path
        
        tokenizer = Tokenizer.from_file(str(json_path))
        
        test_cases = [
            ("영어", "Hello, world! This is a test."),
            ("한국어", "안녕하세요. MOAI-LLM 토크나이저입니다."),
            ("일본어", "こんにちは。トークナイザーのテストです。"),
            ("중국어", "你好。这是分词器测试。"),
            ("코드", "def hello(): print('Hello, World!')"),
            ("금융", "신용카드 결제 한도를 증액하고 싶습니다."),
            ("금융2", "BC카드 포인트 적립률이 어떻게 되나요?"),
            ("숫자", "2024년 1월 15일 오후 3시 30분"),
            ("혼합", "AI 모델의 hidden_size는 3840입니다."),
        ]
        
        print("=" * 70)
        print("📝 Tokenization Test")
        print("=" * 70)
        
        total_chars = 0
        total_tokens = 0
        
        for label, text in test_cases:
            encoded = tokenizer.encode(text)
            tokens = encoded.tokens
            total_chars += len(text)
            total_tokens += len(tokens)
            
            print(f"[{label}] {text}")
            if len(tokens) > 12:
                print(f"   → {len(tokens)} tokens: {tokens[:6]} ... {tokens[-3:]}")
            else:
                print(f"   → {len(tokens)} tokens: {tokens}")
            print()
        
        # 효율성 통계
        chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
        print("=" * 70)
        print("📈 Statistics")
        print("=" * 70)
        print(f"Total characters: {total_chars}")
        print(f"Total tokens: {total_tokens}")
        print(f"Characters per token: {chars_per_token:.2f}")
        print(f"Compression ratio: {chars_per_token:.1f}:1")
        
    else:
        print("⚠️  'tokenizers' library not installed. Skipping tokenization test.")
        print("   Install with: pip install tokenizers")
    
    print()
    print("=" * 70)
    print("✅ Test completed!")
    print("=" * 70)


def compare_tokenizers(tokenizers_dir: str = "tokenizers"):
    """모든 토크나이저 비교"""
    path = Path(tokenizers_dir)
    
    print("=" * 70)
    print("📊 Tokenizer Comparison")
    print("=" * 70)
    print()
    
    tokenizers = []
    
    # 폴더 형태 토크나이저
    for subdir in sorted(path.iterdir()):
        if subdir.is_dir() and (subdir / "tokenizer.json").exists():
            try:
                info = load_tokenizer_info(str(subdir))
                tokenizers.append(info)
            except Exception as e:
                print(f"⚠️  Failed to load {subdir.name}: {e}")
    
    if not tokenizers:
        print("No tokenizers found!")
        return
    
    # 테이블 형태로 출력
    print(f"{'Name':<25} {'Vocab':>12} {'Merges':>12} {'Type':<10}")
    print("-" * 60)
    
    for info in tokenizers:
        name = Path(info['path']).name
        print(f"{name:<25} {info['vocab_size']:>12,} {info['merges']:>12,} {info['model_type']:<10}")
    
    print()
    
    # 최종 토크나이저 하이라이트
    final = next((t for t in tokenizers if Path(t['path']).name == 'moai'), None)
    if final:
        print("=" * 70)
        print(f"🎯 Final Tokenizer: moai")
        print(f"   Vocab: {final['vocab_size']:,} | Merges: {final['merges']:,}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test MOAI tokenizer")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizers/moai",
        help="Path to tokenizer (default: tokenizers/moai)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all tokenizers in the tokenizers directory"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_tokenizers()
    else:
        test_tokenizer(args.tokenizer_path)


if __name__ == "__main__":
    main()

