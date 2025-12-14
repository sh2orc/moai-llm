"""
토크나이저 학습 스크립트 (HuggingFace datasets 지원)

사용법:
    # 새로 학습 (처음부터)
    python train_tokenizer.py \
        --dataset wikipedia \
        --dataset_config 20220301.ko \
        --vocab_size 128000 \
        --output_dir tokenizers/

    # 기존 토크나이저 업데이트 (새 데이터 추가)
    python train_tokenizer.py \
        --base_tokenizer tokenizers/moai_tokenizer.model \
        --dataset pubmed \
        --vocab_size 150000 \
        --output_dir tokenizers/updated/

    # 로컬 txt 파일
    python train_tokenizer.py \
        --input_files data/*.txt \
        --vocab_size 128000 \
        --output_dir tokenizers/

    # 여러 데이터셋 혼합
    python train_tokenizer.py \
        --datasets wikipedia bookcorpus \
        --dataset_configs 20220301.en None \
        --vocab_size 128000 \
        --output_dir tokenizers/
"""

import argparse
import sentencepiece as spm
from pathlib import Path
from typing import List, Optional
import logging
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_and_prepare_text(
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    input_files: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    text_column: str = "text",
    text_columns: Optional[List[str]] = None,
) -> str:
    """
    HuggingFace 데이터셋 또는 로컬 파일에서 텍스트 준비

    Args:
        text_column: 단일 텍스트 컬럼 (기본: "text")
        text_columns: 여러 텍스트 컬럼 (예: ["instruction", "output"])
                     지정 시 text_column 무시하고 모든 컬럼 결합

    Returns:
        임시 텍스트 파일 경로
    """
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')

    if input_files:
        # 로컬 파일 사용
        logger.info(f"📂 Loading from local files: {len(input_files)} files")
        for input_file in input_files:
            with open(input_file, 'r', encoding='utf-8') as f:
                temp_file.write(f.read())
                temp_file.write('\n')

    elif dataset_name:
        # HuggingFace 데이터셋
        logger.info(f"📥 Downloading from HuggingFace: {dataset_name}")

        from datasets import load_dataset

        if dataset_config:
            logger.info(f"   Config: {dataset_config}")
            dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
        else:
            dataset = load_dataset(dataset_name, split="train", streaming=True)

        count = 0
        for item in dataset:
            if max_samples and count >= max_samples:
                break

            # 텍스트 추출
            if text_columns:
                # 여러 컬럼 결합 (instruction-output 등)
                text_parts = []
                for col in text_columns:
                    col_text = item.get(col, "")
                    if col_text:
                        text_parts.append(str(col_text))
                text = " ".join(text_parts)
            else:
                # 단일 컬럼
                text = item.get(text_column, "")

            if text and len(text.strip()) > 50:
                temp_file.write(text)
                temp_file.write('\n')
                count += 1

                if count % 10000 == 0:
                    logger.info(f"   Downloaded {count:,} samples...")

        logger.info(f"✓ Downloaded {count:,} samples")

    else:
        raise ValueError("Either dataset_name or input_files must be provided")

    temp_file.close()
    return temp_file.name


def merge_training_data(base_tokenizer_path: str, new_data_path: str) -> str:
    """
    기존 토크나이저의 학습 데이터와 새 데이터를 병합

    Args:
        base_tokenizer_path: 기존 토크나이저 모델 경로
        new_data_path: 새 학습 데이터 경로

    Returns:
        병합된 데이터 임시 파일 경로
    """
    logger.info("🔄 Merging existing tokenizer data with new data...")

    # 기존 토크나이저로 샘플링한 데이터 생성
    sp = spm.SentencePieceProcessor()
    sp.load(base_tokenizer_path)

    # 기존 어휘의 대표 문장들 추출 (vocab의 일부를 문장으로 변환)
    existing_samples = []
    vocab_size = min(sp.vocab_size(), 10000)  # 최대 10000개 샘플

    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        if piece.startswith('<') or piece.startswith('['):  # 특수 토큰 제외
            continue
        # byte fallback 토큰 제외
        if piece.startswith('<0x'):
            continue
        existing_samples.append(piece)

    # 병합 파일 생성
    merged_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')

    # 기존 어휘 샘플 추가 (50%)
    logger.info(f"   Adding {len(existing_samples):,} samples from existing tokenizer")
    merged_file.write(' '.join(existing_samples))
    merged_file.write('\n')

    # 새 데이터 추가 (50%)
    logger.info(f"   Adding new training data from {new_data_path}")
    with open(new_data_path, 'r', encoding='utf-8') as f:
        merged_file.write(f.read())

    merged_file.close()
    logger.info(f"✓ Data merged successfully")

    return merged_file.name


def train_tokenizer(
    input_file: str,
    vocab_size: int,
    output_dir: str,
    model_prefix: str = "moai_tokenizer",
    character_coverage: float = 0.9995,
    base_tokenizer: Optional[str] = None,
):
    """
    SentencePiece 토크나이저 학습

    Args:
        input_file: 입력 텍스트 파일
        vocab_size: 어휘 크기
        output_dir: 출력 디렉토리
        model_prefix: 모델 파일명 prefix
        character_coverage: 문자 커버리지 (다국어: 0.9995)
        base_tokenizer: 기존 토크나이저 경로 (업데이트 모드)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full_model_prefix = str(output_path / model_prefix)

    # 업데이트 모드인 경우 데이터 병합
    if base_tokenizer:
        logger.info("="*80)
        logger.info("🔄 Updating Existing Tokenizer")
        logger.info("="*80)
        logger.info(f"Base tokenizer: {base_tokenizer}")
        logger.info(f"New vocab size: {vocab_size:,}")
        logger.info("="*80)

        # 기존 토크나이저 정보 출력
        sp_base = spm.SentencePieceProcessor()
        sp_base.load(base_tokenizer)
        logger.info(f"Original vocab size: {sp_base.vocab_size():,}")
        logger.info(f"Vocab increase: +{vocab_size - sp_base.vocab_size():,}")
        logger.info("="*80)

        # 데이터 병합
        merged_input = merge_training_data(base_tokenizer, input_file)
        input_file = merged_input
    else:
        logger.info("="*80)
        logger.info("🔤 Training SentencePiece Tokenizer")
        logger.info("="*80)
        logger.info(f"Vocabulary size: {vocab_size:,}")
        logger.info(f"Character coverage: {character_coverage}")
        logger.info(f"Output: {full_model_prefix}")
        logger.info("="*80)

    # Special tokens (Qwen3 스타일)
    special_tokens = [
        "<|endoftext|>",    # End of text
        "<|im_start|>",     # Instruction message start
        "<|im_end|>",       # Instruction message end
    ]

    # 학습 파라미터
    train_args = {
        "input": input_file,
        "model_prefix": full_model_prefix,
        "model_type": "bpe",
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "num_threads": 16,
        "max_sentence_length": 16384,
        "shuffle_input_sentence": True,
        "add_dummy_prefix": True,
        "remove_extra_whitespaces": True,
        "normalization_rule_name": "nmt_nfkc_cf",
        "pad_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "unk_id": 3,
        "user_defined_symbols": ",".join(special_tokens),
        "split_digits": True,
        "split_by_unicode_script": True,
        "split_by_whitespace": True,
        "split_by_number": True,
        "byte_fallback": True,
    }

    # 학습
    logger.info("🏃 Training...")
    spm.SentencePieceTrainer.train(**train_args)

    # 병합된 임시 파일 삭제
    if base_tokenizer:
        import os
        os.remove(merged_input)

    # 검증
    sp = spm.SentencePieceProcessor()
    sp.load(f"{full_model_prefix}.model")

    logger.info("="*80)
    logger.info("✅ Tokenizer trained successfully!")
    logger.info("="*80)
    logger.info(f"Vocabulary size: {sp.vocab_size():,}")
    logger.info(f"BOS: {sp.id_to_piece(sp.bos_id())} (id={sp.bos_id()})")
    logger.info(f"EOS: {sp.id_to_piece(sp.eos_id())} (id={sp.eos_id()})")
    logger.info(f"PAD: {sp.id_to_piece(sp.pad_id())} (id={sp.pad_id()})")
    logger.info(f"UNK: {sp.id_to_piece(sp.unk_id())} (id={sp.unk_id()})")

    # 테스트
    test_texts = [
        "Hello, world! This is a test.",
        "안녕하세요. 토크나이저 테스트입니다.",
        "print('Hello, World!')",
    ]

    logger.info("="*80)
    logger.info("🧪 Tokenization Test")
    logger.info("="*80)
    for text in test_texts:
        tokens = sp.encode(text, out_type=str)
        logger.info(f"Text: {text}")
        logger.info(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        logger.info(f"Count: {len(tokens)}")
        logger.info("-"*80)

    logger.info("="*80)
    logger.info(f"📁 Saved to: {full_model_prefix}.model")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece Tokenizer")

    # 데이터 소스
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        type=str,
        help="HuggingFace dataset name (e.g., wikipedia)"
    )
    group.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="Local text files"
    )

    # 데이터셋 옵션
    parser.add_argument(
        "--dataset_config",
        type=str,
        help="Dataset config (e.g., 20220301.ko)"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Text column name (default: text)"
    )
    parser.add_argument(
        "--text_columns",
        type=str,
        nargs="+",
        default=None,
        help="Multiple text columns to combine (e.g., instruction output). Overrides --text_column"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples from dataset (default: all)"
    )

    # 토크나이저 설정
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=128000,
        help="Vocabulary size (default: 128000)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tokenizers",
        help="Output directory (default: tokenizers)"
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        default="moai_tokenizer",
        help="Model file prefix (default: moai_tokenizer)"
    )
    parser.add_argument(
        "--character_coverage",
        type=float,
        default=0.9995,
        help="Character coverage (default: 0.9995 for multilingual)"
    )

    # 업데이트 모드
    parser.add_argument(
        "--base_tokenizer",
        type=str,
        default=None,
        help="Base tokenizer model path to update (optional)"
    )

    args = parser.parse_args()

    # 1. 텍스트 준비
    logger.info("📚 Preparing training data...")
    temp_file = download_and_prepare_text(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        input_files=args.input_files,
        max_samples=args.max_samples,
        text_column=args.text_column,
        text_columns=args.text_columns,
    )

    # 2. 토크나이저 학습
    train_tokenizer(
        input_file=temp_file,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        model_prefix=args.model_prefix,
        character_coverage=args.character_coverage,
        base_tokenizer=args.base_tokenizer,
    )

    # 3. 임시 파일 삭제
    import os
    os.remove(temp_file)

    logger.info("\n✨ All done!")


if __name__ == "__main__":
    main()
