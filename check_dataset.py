"""
데이터셋 정보 확인 스크립트

Usage:
    python check_dataset.py wikipedia
    python check_dataset.py allenai/c4
    python check_dataset.py BCCard/BCCard-Finance-Kor-QnA
"""

import argparse
from datasets import get_dataset_config_names, load_dataset_builder

def check_dataset(dataset_name: str):
    """데이터셋 정보 출력"""

    print("="*80)
    print(f"Dataset: {dataset_name}")
    print("="*80)

    # 1. Config 목록 확인
    try:
        configs = get_dataset_config_names(dataset_name)
        if configs:
            print(f"\n📋 Available Configs ({len(configs)} total):")
            print("-"*80)

            # 한국어/영어 관련만 필터링
            ko_configs = [c for c in configs if 'ko' in c.lower()]
            en_configs = [c for c in configs if 'en' in c.lower()]

            if ko_configs:
                print(f"Korean configs: {ko_configs[:5]}")
            if en_configs:
                print(f"English configs: {en_configs[:5]}")

            # 처음 10개만 표시
            print(f"\nFirst 10 configs:")
            for i, config in enumerate(configs[:10]):
                print(f"  {i+1}. {config}")

            if len(configs) > 10:
                print(f"  ... and {len(configs) - 10} more")
        else:
            print("\n📋 No config needed (use dataset_config=None)")

    except Exception as e:
        print(f"\n⚠️ Could not get configs: {e}")
        print("This dataset might not need a config.")

    # 2. 데이터셋 정보
    try:
        builder = load_dataset_builder(dataset_name)
        print(f"\n📝 Dataset Info:")
        print("-"*80)
        print(f"Description: {builder.info.description[:200]}...")

        if builder.info.features:
            print(f"\n🔑 Features (columns):")
            for name, feature in list(builder.info.features.items())[:5]:
                print(f"  - {name}: {feature}")

    except Exception as e:
        print(f"\n⚠️ Could not get dataset info: {e}")

    # 3. 사용 예시
    print(f"\n💡 Usage Examples:")
    print("-"*80)

    if configs and len(configs) > 0:
        # Config가 있는 경우
        example_config = ko_configs[0] if ko_configs else configs[0]

        print(f"# Tokenizer training")
        print(f"python train_tokenizer.py \\")
        print(f"    --dataset {dataset_name} \\")
        print(f"    --dataset_config {example_config} \\")
        print(f"    --output_dir tokenizers/")

        print(f"\n# Pretraining")
        print(f"python train.py \\")
        print(f"    --mode pretrain \\")
        print(f"    --dataset {dataset_name} \\")
        print(f"    --dataset_config {example_config} \\")
        print(f"    --output_dir outputs/pretrain")
    else:
        # Config가 없는 경우
        print(f"# Tokenizer training (no config needed)")
        print(f"python train_tokenizer.py \\")
        print(f"    --dataset {dataset_name} \\")
        print(f"    --output_dir tokenizers/")

        print(f"\n# SFT (for Q&A datasets)")
        print(f"python train.py \\")
        print(f"    --mode sft \\")
        print(f"    --dataset {dataset_name} \\")
        print(f"    --pretrained_model outputs/pretrain/final_model \\")
        print(f"    --output_dir outputs/sft")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Check HuggingFace dataset info")
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., wikipedia)")
    args = parser.parse_args()

    check_dataset(args.dataset_name)


if __name__ == "__main__":
    main()
