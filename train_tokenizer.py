"""
토크나이저 학습 스크립트 (HuggingFace Tokenizers - Rust 기반, 초고속)

🚀 SentencePiece 대비 10~100배 빠른 학습 속도!

사용법:
    # Step 1: 다국어 64K
    python train_tokenizer.py \
        --multilingual ko en ja zh \
        --vocab_size 64000 \
        --max_samples_per_lang 60000 \
        --turbo \
        --output_dir tokenizers/ \
        --model_prefix moai_multilingual

    # Step 2: Alpaca +16K → 80K
    python train_tokenizer.py \
        --base_tokenizer tokenizers/moai_multilingual \
        --dataset unoooo/alpaca-korean \
        --vocab_size 80000 \
        --max_samples_per_lang 30000 \
        --turbo \
        --output_dir tokenizers/ \
        --model_prefix moai_alpaca


    # Step 3: 금융 +16K → 96K
    python train_tokenizer.py \
        --base_tokenizer tokenizers/moai_alpaca \
        --dataset Mineru/kor-open-finance \
        --vocab_size 96000 \
        --max_samples 30000 \
        --turbo \
        --output_dir tokenizers/ \
        --model_prefix moai_finance

    # Step 4: 금융 +16K → 128K
    python train_tokenizer.py \
        --base_tokenizer tokenizers/moai_finance \
        --dataset BCCard/BCAI-Finance-Kor \
        --vocab_size 128000 \
        --max_samples 100000 \
        --turbo \
        --output_dir tokenizers/ \
        --model_prefix moai_finance_bccard

----------------------------------------------------------

    # 로컬 txt 파일
    python train_tokenizer.py \
        --input_files data/*.txt \
        --vocab_size 128000 \
        --output_dir tokenizers/

    # 빠른 테스트 (샘플 제한)
    python train_tokenizer.py \
        --dataset wikimedia/wikipedia \
        --dataset_config 20231101.ko \
        --vocab_size 32000 \
        --max_samples 100000 \
        --output_dir tokenizers/test

    # 🚀 FAST 모드 (대용량에서도 빠르게!)
    python train_tokenizer.py \
        --multilingual ko en ja zh \
        --vocab_size 128000 \
        --max_samples_per_lang 500000 \
        --fast \
        --output_dir tokenizers/

    # ⚡ ULTRAFAST 모드 (Unigram 알고리즘, 최대 속도!)
    python train_tokenizer.py \
        --multilingual ko en ja zh \
        --vocab_size 128000 \
        --max_samples_per_lang 500000 \
        --ultrafast \
        --output_dir tokenizers/
"""

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import List, Optional, Iterator, Dict
import logging
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# tokenizers 병렬 처리 경고 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def signal_handler(sig, frame):
    """Ctrl+C 시그널 핸들러"""
    print("\n\n⚠️  Interrupted by user. Exiting...")
    sys.exit(1)

# Ctrl+C 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 언어별 Wikipedia config 매핑
WIKI_LANG_CONFIG = {
    "ko": "20231101.ko",    # 한국어
    "en": "20231101.en",    # 영어
    "ja": "20231101.ja",    # 일본어
    "zh": "20231101.zh",    # 중국어
    "de": "20231101.de",    # 독일어
    "fr": "20231101.fr",    # 프랑스어
    "es": "20231101.es",    # 스페인어
    "ru": "20231101.ru",    # 러시아어
    "pt": "20231101.pt",    # 포르투갈어
    "it": "20231101.it",    # 이탈리아어
    "vi": "20231101.vi",    # 베트남어
    "th": "20231101.th",    # 태국어
    "ar": "20231101.ar",    # 아랍어
}


def text_iterator(
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    input_files: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    text_column: str = "text",
    text_columns: Optional[List[str]] = None,
    batch_size: int = 1000,
) -> Iterator[List[str]]:
    """
    텍스트 배치 이터레이터 (메모리 효율적)
    
    HuggingFace Tokenizers는 이터레이터를 직접 받아서 학습 가능
    → 임시 파일 불필요, 메모리 효율적
    """
    
    if input_files:
        # 로컬 파일에서 읽기
        logger.info(f"📂 Loading from local files: {len(input_files)} files")
        batch = []
        count = 0
        
        for input_file in input_files:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 50:
                        batch.append(line)
                        count += 1
                        
                        if max_samples and count >= max_samples:
                            if batch:
                                yield batch
                            logger.info(f"✓ Loaded {count:,} samples")
                            return
                        
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                            
                            if count % 100000 == 0:
                                logger.info(f"   Loaded {count:,} samples...")
        
        if batch:
            yield batch
        logger.info(f"✓ Loaded {count:,} samples")
    
    elif dataset_name:
        # HuggingFace 데이터셋
        logger.info(f"📥 Streaming from HuggingFace: {dataset_name}")
        
        from datasets import load_dataset
        
        if dataset_config:
            logger.info(f"   Config: {dataset_config}")
            dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
        else:
            dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        batch = []
        count = 0
        detected_columns = None
        
        # 자동 컬럼 감지를 위한 일반적인 텍스트 컬럼 이름들
        common_text_columns = [
            "text", "content", "sentence", "document", "body",
            "instruction", "input", "output", "response", "answer", "question",
            "prompt", "completion", "context", "passage",
            "query", "reply", "message", "description",
        ]
        
        for item in dataset:
            # 첫 아이템에서 컬럼 자동 감지
            if detected_columns is None and not text_columns:
                if text_column in item and item.get(text_column):
                    detected_columns = [text_column]
                else:
                    # 일반적인 텍스트 컬럼 찾기
                    detected_columns = []
                    for col in common_text_columns:
                        if col in item and item.get(col):
                            detected_columns.append(col)
                    
                    # 여전히 없으면 모든 문자열 필드 사용
                    if not detected_columns:
                        detected_columns = [k for k, v in item.items() if isinstance(v, str) and len(v) > 10]
                    
                    if detected_columns:
                        logger.info(f"   Auto-detected columns: {detected_columns}")
            
            # 텍스트 추출
            cols_to_use = text_columns or detected_columns or [text_column]
            if len(cols_to_use) > 1 or text_columns:
                text_parts = []
                for col in cols_to_use:
                    col_text = item.get(col, "")
                    if col_text:
                        text_parts.append(str(col_text))
                text = " ".join(text_parts)
            else:
                text = item.get(cols_to_use[0], "") if cols_to_use else ""
            
            if text and len(text.strip()) > 50:
                batch.append(text)
                count += 1
                
                if max_samples and count >= max_samples:
                    if batch:
                        yield batch
                    logger.info(f"✓ Downloaded {count:,} samples")
                    return
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
                    
                    if count % 100000 == 0:
                        logger.info(f"   Downloaded {count:,} samples...")
        
        if batch:
            yield batch
        logger.info(f"✓ Downloaded {count:,} samples")
    
    else:
        raise ValueError("Either dataset_name or input_files must be provided")


def _load_language_data(
    lang: str,
    config: str,
    max_samples: Optional[int],
    output_queue: Queue,
    stop_event: threading.Event
) -> Dict:
    """
    단일 언어 데이터를 로드하는 워커 함수 (별도 스레드에서 실행)
    """
    from datasets import load_dataset
    
    result = {"lang": lang, "count": 0, "error": None}
    
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            config,
            split="train",
            streaming=True
        )
        
        count = 0
        for item in dataset:
            if stop_event.is_set():
                break
                
            text = item.get("text", "")
            if text and len(text.strip()) > 50:
                output_queue.put((lang, text))
                count += 1
                
                if count % 10000 == 0:
                    logger.info(f"   [{lang.upper()}] {count:,} samples...")
                
                if max_samples and count >= max_samples:
                    break
        
        result["count"] = count
        
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"⚠️  Failed to load {lang}: {e}")
    
    # 완료 신호
    output_queue.put((lang, None))
    return result


def multilingual_text_iterator(
    languages: List[str],
    max_samples_per_lang: Optional[int] = None,
    batch_size: int = 1000,
    parallel: bool = True,
) -> Iterator[List[str]]:
    """
    다국어 텍스트 이터레이터 - 여러 언어의 Wikipedia를 병렬 로딩!
    
    Args:
        languages: 언어 코드 리스트 (예: ["ko", "en", "ja", "zh"])
        max_samples_per_lang: 언어별 최대 샘플 수
        batch_size: 배치 크기
        parallel: 병렬 로딩 활성화 (기본: True)
    """
    from datasets import load_dataset
    
    valid_languages = [lang for lang in languages if lang in WIKI_LANG_CONFIG]
    if not valid_languages:
        raise ValueError(f"No valid languages found. Available: {list(WIKI_LANG_CONFIG.keys())}")
    
    logger.info("="*80)
    logger.info("🌍 Multilingual Wikipedia Loading" + (" (PARALLEL)" if parallel else ""))
    logger.info("="*80)
    logger.info(f"Languages: {', '.join(valid_languages)}")
    logger.info(f"Max samples per language: {max_samples_per_lang or 'unlimited'}")
    if parallel:
        logger.info(f"🚀 Loading {len(valid_languages)} languages in PARALLEL!")
    logger.info("="*80)
    
    if parallel and len(valid_languages) > 1:
        # ========== 병렬 로딩 모드 ==========
        data_queue = Queue(maxsize=50000)  # 메모리 제한을 위한 최대 큐 크기
        stop_event = threading.Event()
        
        # 각 언어별 스레드 시작
        futures = []
        with ThreadPoolExecutor(max_workers=len(valid_languages)) as executor:
            for lang in valid_languages:
                config = WIKI_LANG_CONFIG[lang]
                logger.info(f"📥 [{lang.upper()}] Starting parallel download ({config})...")
                future = executor.submit(
                    _load_language_data,
                    lang, config, max_samples_per_lang, data_queue, stop_event
                )
                futures.append((lang, future))
            
            # 데이터 수집 및 yield
            batch = []
            total_count = 0
            lang_counts = {lang: 0 for lang in valid_languages}
            completed_langs = set()
            
            try:
                while len(completed_langs) < len(valid_languages):
                    try:
                        lang, text = data_queue.get(timeout=1.0)
                        
                        if text is None:
                            # 해당 언어 완료
                            completed_langs.add(lang)
                            logger.info(f"   [{lang.upper()}] ✓ {lang_counts[lang]:,} samples complete")
                            continue
                        
                        batch.append(text)
                        total_count += 1
                        lang_counts[lang] += 1
                        
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                            
                    except Exception:
                        # Queue 타임아웃 - 계속 진행
                        pass
                
                # 남은 배치 yield
                if batch:
                    yield batch
                    
            except GeneratorExit:
                stop_event.set()
                raise
        
        logger.info(f"\n🌍 Total: {total_count:,} samples from {len(valid_languages)} languages (PARALLEL)")
        for lang, count in lang_counts.items():
            logger.info(f"   [{lang.upper()}] {count:,}")
    
    else:
        # ========== 순차 로딩 모드 (단일 언어 또는 parallel=False) ==========
        total_count = 0
        
        for lang in valid_languages:
            config = WIKI_LANG_CONFIG[lang]
            logger.info(f"\n📥 [{lang.upper()}] Loading wikimedia/wikipedia ({config})...")
            
            try:
                dataset = load_dataset(
                    "wikimedia/wikipedia",
                    config,
                    split="train",
                    streaming=True
                )
            except Exception as e:
                logger.warning(f"⚠️  Failed to load {lang}: {e}")
                continue
            
            batch = []
            count = 0
            
            for item in dataset:
                text = item.get("text", "")
                
                if text and len(text.strip()) > 50:
                    batch.append(text)
                    count += 1
                    total_count += 1
                    
                    if max_samples_per_lang and count >= max_samples_per_lang:
                        if batch:
                            yield batch
                        logger.info(f"   [{lang.upper()}] ✓ {count:,} samples (limit reached)")
                        break
                    
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                        
                        if count % 100000 == 0:
                            logger.info(f"   [{lang.upper()}] Downloaded {count:,} samples...")
            
            if batch:
                yield batch
            
            if not (max_samples_per_lang and count >= max_samples_per_lang):
                logger.info(f"   [{lang.upper()}] ✓ {count:,} samples (complete)")
        
        logger.info(f"\n🌍 Total: {total_count:,} samples from {len(valid_languages)} languages")


def extend_tokenizer(
    base_tokenizer_path: str,
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    input_files: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    text_column: str = "text",
    text_columns: Optional[List[str]] = None,
    new_vocab_size: int = 150000,
    output_dir: str = "tokenizers",
    model_prefix: str = "moai_tokenizer_extended",
    min_frequency: int = 2,
    languages: Optional[List[str]] = None,
    max_samples_per_lang: Optional[int] = None,
    fast_mode: bool = False,
    ultrafast_mode: bool = False,
    turbo_mode: bool = False,
    parallel: bool = True,
):
    """
    기존 토크나이저를 확장하여 새 토큰 추가 학습
    
    핵심: 기존 vocab과 merges를 유지하면서 새 토큰만 추가!
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    from tokenizers.normalizers import NFKC, Sequence
    from transformers import PreTrainedTokenizerFast
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 기존 토크나이저 로드
    base_path = Path(base_tokenizer_path)
    
    if base_path.is_file() and base_path.suffix == '.json':
        tokenizer_json_path = base_path
    elif base_path.is_dir():
        tokenizer_json_path = base_path / "tokenizer.json"
        if not tokenizer_json_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {base_path}")
    else:
        raise FileNotFoundError(f"Tokenizer not found: {base_tokenizer_path}")
    
    # JSON 직접 로드하여 vocab과 merges 추출
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    # 기존 vocab과 merges 추출
    model_data = tokenizer_data.get("model", {})
    existing_vocab = model_data.get("vocab", {})
    existing_merges = model_data.get("merges", [])
    
    original_vocab_size = len(existing_vocab)
    original_merges_count = len(existing_merges)
    
    logger.info("="*80)
    logger.info("🔄 Extending Existing Tokenizer (Preserving Vocab & Merges)")
    logger.info("="*80)
    logger.info(f"Base tokenizer: {base_tokenizer_path}")
    logger.info(f"Original vocab size: {original_vocab_size:,}")
    logger.info(f"Original merges: {original_merges_count:,}")
    logger.info(f"Target vocab size: {new_vocab_size:,}")
    logger.info(f"Vocab to add: +{new_vocab_size - original_vocab_size:,}")
    logger.info("="*80)
    
    if new_vocab_size <= original_vocab_size:
        raise ValueError(f"new_vocab_size ({new_vocab_size}) must be greater than original ({original_vocab_size})")
    
    # 추가할 vocab 수 계산
    vocab_to_add = new_vocab_size - original_vocab_size
    
    # 모드별 최적화
    if ultrafast_mode:
        min_frequency = max(min_frequency, 3)
        limit_alphabet = 10000
        logger.info("⚡⚡ ULTRAFAST MODE ENABLED")
    elif turbo_mode:
        min_frequency = max(min_frequency, 10)
        limit_alphabet = 5000
        logger.info("🚀 TURBO MODE ENABLED")
    elif fast_mode:
        min_frequency = max(min_frequency, 5)
        limit_alphabet = 10000
        logger.info("⚡ FAST MODE ENABLED")
    else:
        limit_alphabet = None
    
    # 새 데이터에서 추가 토큰 학습을 위한 임시 토크나이저
    temp_tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    temp_tokenizer.normalizer = Sequence([NFKC()])
    temp_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    temp_tokenizer.decoder = decoders.ByteLevel()
    
    special_tokens = [
        "<pad>", "<s>", "</s>", "<unk>",
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    ]
    
    # 추가 토큰만 학습 (추가할 양 + 버퍼)
    # 핵심: 전체 크기가 아닌, 필요한 만큼만 학습!
    # merge 결과 토큰도 고려해서 약간의 버퍼 추가 (1.5배)
    train_vocab_size = int(vocab_to_add * 1.5) + len(special_tokens)
    logger.info(f"   Training for ~{train_vocab_size:,} new tokens (buffer included)")
    
    trainer_kwargs = {
        "vocab_size": train_vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "show_progress": True,
        "initial_alphabet": pre_tokenizers.ByteLevel.alphabet(),
    }
    
    if limit_alphabet:
        trainer_kwargs["limit_alphabet"] = limit_alphabet
    
    trainer = trainers.BpeTrainer(**trainer_kwargs)
    
    logger.info("🏃 Training additional tokens from new data...")
    start_time = time.time()
    
    def get_training_corpus():
        # 1. 기존 토큰들을 텍스트로 포함 (merges 학습에 도움)
        existing_tokens = list(existing_vocab.keys())
        vocab_sentences = [t for t in existing_tokens if not t.startswith('<') and len(t) > 1]
        batch_size = 1000
        for i in range(0, len(vocab_sentences), batch_size):
            yield vocab_sentences[i:i+batch_size]
        logger.info(f"   Included {len(vocab_sentences):,} existing tokens as seed")
        
        # 2. 새 데이터
        if languages:
            for batch in multilingual_text_iterator(languages, max_samples_per_lang, parallel=parallel):
                yield batch
        else:
            for batch in text_iterator(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                input_files=input_files,
                max_samples=max_samples,
                text_column=text_column,
                text_columns=text_columns,
            ):
                yield batch
    
    temp_tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    elapsed = time.time() - start_time
    logger.info(f"⏱️  Additional training completed in {elapsed:.1f} seconds!")
    
    # 새로 학습된 vocab과 merges 추출
    temp_json = json.loads(temp_tokenizer.to_str())
    new_trained_vocab = temp_json["model"]["vocab"]
    new_trained_merges = temp_json["model"]["merges"]
    
    logger.info(f"   New training produced {len(new_trained_vocab):,} tokens, {len(new_trained_merges):,} merges")
    
    # ========== 핵심: 기존 vocab/merges 유지 + 새 토큰 추가 ==========
    logger.info("🔗 Merging vocabularies (preserving original)...")
    
    # 최종 vocab: 기존 vocab을 그대로 유지
    final_vocab = dict(existing_vocab)
    next_id = max(existing_vocab.values()) + 1
    
    # 최종 merges: 기존 merges 유지 + 새 merges 추가
    # existing_merges는 리스트의 리스트 형태일 수 있음 ([['a', 'b'], ...])
    # new_trained_merges는 문자열 형태 (['a b', ...])
    final_merges = list(existing_merges)
    
    # 기존 merges를 튜플 형태로 변환하여 set 생성 (리스트는 해시 불가)
    def normalize_merge(m):
        """merge를 튜플 형태로 정규화"""
        if isinstance(m, list):
            return tuple(m)
        elif isinstance(m, str):
            return tuple(m.split(" "))
        return m
    
    existing_merges_set = set(normalize_merge(m) for m in existing_merges)
    
    # 새로 학습된 토큰 중 기존에 없는 것만 추가 (merge 결과 토큰 포함해서 총량 제한)
    total_added = 0
    
    # 1단계: 새로 학습된 vocab에서 직접 토큰 추가
    added_tokens = 0
    for token in new_trained_vocab:
        if token not in final_vocab and total_added < vocab_to_add:
            final_vocab[token] = next_id
            next_id += 1
            added_tokens += 1
            total_added += 1
    
    logger.info(f"   Added {added_tokens:,} new tokens from vocab")
    
    # 2단계: merge 추가 (결과 토큰도 총량 제한 내에서만 추가)
    added_merges = 0
    added_merge_tokens = 0
    skipped_merges = 0
    
    for merge in new_trained_merges:
        merge_tuple = normalize_merge(merge)
        if merge_tuple not in existing_merges_set:
            parts = list(merge_tuple) if isinstance(merge_tuple, tuple) else merge.split(" ")
            if len(parts) == 2:
                part1, part2 = parts[0], parts[1]
                merged_token = part1 + part2
                
                # 입력 토큰들이 vocab에 있어야 함
                if part1 not in final_vocab or part2 not in final_vocab:
                    continue
                
                # 결과 토큰이 vocab에 없으면 추가 (총량 제한 체크!)
                if merged_token not in final_vocab:
                    if total_added >= vocab_to_add:
                        # 제한 초과: 이 merge는 건너뛰기
                        skipped_merges += 1
                        continue
                    final_vocab[merged_token] = next_id
                    next_id += 1
                    added_merge_tokens += 1
                    total_added += 1
                
                # merge 추가 (기존 merges 형태에 맞춰서)
                if existing_merges and isinstance(existing_merges[0], list):
                    final_merges.append(list(merge_tuple))
                else:
                    final_merges.append(merge)
                existing_merges_set.add(merge_tuple)
                added_merges += 1
    
    logger.info(f"   Added {added_merges:,} new merges")
    if added_merge_tokens > 0:
        logger.info(f"   Added {added_merge_tokens:,} merge result tokens to vocab")
    if skipped_merges > 0:
        logger.info(f"   Skipped {skipped_merges:,} merges (vocab limit reached)")
    logger.info(f"   Total tokens added: {total_added:,} / {vocab_to_add:,}")
    
    # 최종 토크나이저 JSON 생성
    final_tokenizer_data = dict(tokenizer_data)
    final_tokenizer_data["model"]["vocab"] = final_vocab
    final_tokenizer_data["model"]["merges"] = final_merges
    
    # JSON으로 저장 후 로드
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(final_tokenizer_data, f, ensure_ascii=False)
        temp_path = f.name
    
    final_tokenizer = Tokenizer.from_file(temp_path)
    os.unlink(temp_path)  # 임시 파일 삭제
    
    # Post-processor 설정
    final_tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[
            ("<s>", final_tokenizer.token_to_id("<s>")),
            ("</s>", final_tokenizer.token_to_id("</s>")),
        ],
    )
    
    # 저장
    tokenizer_path = output_path / f"{model_prefix}.json"
    final_tokenizer.save(str(tokenizer_path))
    
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=final_tokenizer,
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
        clean_up_tokenization_spaces=False,
    )
    hf_tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    })
    hf_tokenizer.save_pretrained(str(output_path / model_prefix))
    
    # 검증
    final_vocab_size = final_tokenizer.get_vocab_size()
    
    logger.info("="*80)
    logger.info("✅ Tokenizer extended successfully!")
    logger.info("="*80)
    logger.info(f"Original vocab: {original_vocab_size:,}")
    logger.info(f"Final vocab: {final_vocab_size:,}")
    logger.info(f"Added tokens: +{final_vocab_size - original_vocab_size:,}")
    logger.info(f"Original merges: {original_merges_count:,}")
    logger.info(f"Final merges: {len(final_merges):,}")
    
    # 다국어 테스트
    _test_tokenizer(final_tokenizer)
    
    logger.info("="*80)
    logger.info(f"📁 Saved to:")
    logger.info(f"   - {tokenizer_path}")
    logger.info(f"   - {output_path / model_prefix}/")
    logger.info("="*80)
    
    return final_tokenizer


def train_tokenizer(
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    input_files: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    text_column: str = "text",
    text_columns: Optional[List[str]] = None,
    vocab_size: int = 128000,
    output_dir: str = "tokenizers",
    model_prefix: str = "moai_tokenizer",
    min_frequency: int = 2,
    languages: Optional[List[str]] = None,
    max_samples_per_lang: Optional[int] = None,
    fast_mode: bool = False,
    ultrafast_mode: bool = False,
    turbo_mode: bool = False,
    parallel: bool = True,
):
    """
    HuggingFace Tokenizers로 토크나이저 학습 (Rust 기반, 초고속)
    
    Args:
        fast_mode: BPE + min_frequency=5, limit_alphabet=10000 (10배 빠름)
        turbo_mode: BPE + min_frequency=10, limit_alphabet=5000 (20배 빠름)
        ultrafast_mode: Unigram 알고리즘 사용 (50배 빠름, Compute merges 없음)
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    from tokenizers.normalizers import NFKC, Sequence
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 알고리즘 선택
    use_unigram = ultrafast_mode
    
    # 모드별 최적화 (turbo > fast > 일반)
    if ultrafast_mode:
        min_frequency = max(min_frequency, 3)
        limit_alphabet = 10000
        logger.info("="*80)
        logger.info("⚡⚡ ULTRAFAST MODE - Unigram algorithm (NO Compute merges!)")
        logger.info("="*80)
    elif turbo_mode:
        min_frequency = max(min_frequency, 10)  # 10회 이상만!
        limit_alphabet = 5000  # 더 작은 알파벳
        logger.info("="*80)
        logger.info("🚀 TURBO MODE - Aggressive BPE optimization")
        logger.info("="*80)
    elif fast_mode:
        min_frequency = max(min_frequency, 5)
        limit_alphabet = 10000
        logger.info("="*80)
        logger.info("⚡ FAST MODE - Optimized BPE")
        logger.info("="*80)
    else:
        limit_alphabet = None
    
    logger.info("="*80)
    if languages:
        logger.info("🌍 Training Multilingual Tokenizer")
        logger.info(f"Languages: {', '.join(languages)}")
    else:
        logger.info("🚀 Training Tokenizer (HuggingFace Tokenizers - Rust)")
    logger.info("="*80)
    logger.info(f"Algorithm: {'Unigram' if use_unigram else 'BPE'}")
    logger.info(f"Vocabulary size: {vocab_size:,}")
    logger.info(f"Min frequency: {min_frequency}")
    if limit_alphabet:
        logger.info(f"Limit alphabet: {limit_alphabet:,}")
    logger.info(f"Output: {output_path / model_prefix}")
    logger.info("="*80)
    
    special_tokens = [
        "<pad>", "<s>", "</s>", "<unk>",
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    ]
    
    if use_unigram:
        # Unigram 모델 (훨씬 빠름!)
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.normalizer = Sequence([NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
    else:
        # BPE 모델
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = Sequence([NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        
        trainer_kwargs = {
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "special_tokens": special_tokens,
            "show_progress": True,
            "initial_alphabet": pre_tokenizers.ByteLevel.alphabet(),
        }
        
        if limit_alphabet:
            trainer_kwargs["limit_alphabet"] = limit_alphabet
        
        trainer = trainers.BpeTrainer(**trainer_kwargs)
    
    logger.info("🏃 Training... (this is FAST with Rust!)")
    start_time = time.time()
    
    def get_training_corpus():
        if languages:
            # 다국어 모드
            for batch in multilingual_text_iterator(languages, max_samples_per_lang, parallel=parallel):
                yield batch
        else:
            # 단일 데이터셋 모드
            for batch in text_iterator(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                input_files=input_files,
                max_samples=max_samples,
                text_column=text_column,
                text_columns=text_columns,
            ):
                yield batch
    
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    elapsed = time.time() - start_time
    logger.info(f"⏱️  Training completed in {elapsed:.1f} seconds!")
    
    # Post-processor
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    
    # 저장
    tokenizer_path = output_path / f"{model_prefix}.json"
    tokenizer.save(str(tokenizer_path))
    
    from transformers import PreTrainedTokenizerFast
    
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
        clean_up_tokenization_spaces=False,
    )
    hf_tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    })
    hf_tokenizer.save_pretrained(str(output_path / model_prefix))
    
    # 검증
    logger.info("="*80)
    logger.info("✅ Tokenizer trained successfully!")
    logger.info("="*80)
    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    logger.info(f"PAD: <pad> (id={tokenizer.token_to_id('<pad>')})")
    logger.info(f"BOS: <s> (id={tokenizer.token_to_id('<s>')})")
    logger.info(f"EOS: </s> (id={tokenizer.token_to_id('</s>')})")
    logger.info(f"UNK: <unk> (id={tokenizer.token_to_id('<unk>')})")
    
    # 다국어 테스트
    _test_tokenizer(tokenizer)
    
    logger.info("="*80)
    logger.info(f"📁 Saved to:")
    logger.info(f"   - {tokenizer_path} (raw tokenizer)")
    logger.info(f"   - {output_path / model_prefix}/ (HuggingFace format)")
    logger.info("="*80)
    
    return tokenizer


def _test_tokenizer(tokenizer):
    """다국어 토크나이저 테스트"""
    test_texts = [
        ("English", "Hello, world! This is a test."),
        ("한국어", "안녕하세요. 토크나이저 테스트입니다."),
        ("日本語", "こんにちは。トークナイザーのテストです。"),
        ("中文", "你好。这是分词器测试。"),
        ("Code", "print('Hello, World!')"),
    ]
    
    logger.info("="*80)
    logger.info("🧪 Multilingual Tokenization Test")
    logger.info("="*80)
    
    for lang, text in test_texts:
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens[:12]
        logger.info(f"[{lang}] {text}")
        logger.info(f"   Tokens: {tokens}{'...' if len(encoded.tokens) > 12 else ''}")
        logger.info(f"   Count: {len(encoded.tokens)}")


def main():
    parser = argparse.ArgumentParser(
        description="Train BPE Tokenizer (HuggingFace Tokenizers - Rust, 10-100x faster!)"
    )
    
    # 데이터 소스 (3가지 중 하나)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        type=str,
        help="HuggingFace dataset name (e.g., wikimedia/wikipedia)"
    )
    group.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="Local text files"
    )
    group.add_argument(
        "--multilingual",
        type=str,
        nargs="+",
        metavar="LANG",
        help="Language codes for multilingual training (e.g., ko en ja zh)"
    )
    
    # 데이터셋 옵션
    parser.add_argument(
        "--dataset_config",
        type=str,
        help="Dataset config (e.g., 20231101.ko)"
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
        help="Multiple text columns to combine"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples from dataset (default: all)"
    )
    parser.add_argument(
        "--max_samples_per_lang",
        type=int,
        default=None,
        help="Max samples per language for multilingual mode"
    )
    
    # 토크나이저 설정
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=128000,
        help="Vocabulary size (default: 128000)"
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum token frequency (default: 2)"
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
    
    # 추가 학습 모드
    parser.add_argument(
        "--base_tokenizer",
        type=str,
        default=None,
        help="Path to base tokenizer for extension"
    )
    
    # 최적화 옵션
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: higher min_frequency, limited alphabet (10x faster)"
    )
    parser.add_argument(
        "--ultrafast",
        action="store_true",
        help="Ultrafast mode: Unigram algorithm instead of BPE (50x faster, no 'Compute merges')"
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Turbo BPE mode: min_frequency=10, limit_alphabet=5000 (20x faster BPE)"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        dest="no_parallel",
        help="Disable parallel loading for multilingual mode (sequential, one language at a time)"
    )
    
    args = parser.parse_args()
    
    logger.info("📚 Starting tokenizer training...")
    logger.info("🚀 Using HuggingFace Tokenizers (Rust-based, blazing fast!)")
    
    if args.base_tokenizer:
        # 확장 모드
        extend_tokenizer(
            base_tokenizer_path=args.base_tokenizer,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            input_files=args.input_files,
            max_samples=args.max_samples,
            text_column=args.text_column,
            text_columns=args.text_columns,
            new_vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            output_dir=args.output_dir,
            model_prefix=args.model_prefix,
            languages=args.multilingual,
            max_samples_per_lang=args.max_samples_per_lang,
            fast_mode=args.fast,
            ultrafast_mode=args.ultrafast,
            turbo_mode=args.turbo,
            parallel=not args.no_parallel,
        )
    else:
        # 새로 학습
        train_tokenizer(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            input_files=args.input_files,
            max_samples=args.max_samples,
            text_column=args.text_column,
            text_columns=args.text_columns,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            output_dir=args.output_dir,
            model_prefix=args.model_prefix,
            languages=args.multilingual,
            max_samples_per_lang=args.max_samples_per_lang,
            fast_mode=args.fast,
            ultrafast_mode=args.ultrafast,
            turbo_mode=args.turbo,
            parallel=not args.no_parallel,
        )
    
    logger.info("\n✨ All done!")


if __name__ == "__main__":
    main()
