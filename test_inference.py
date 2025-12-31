"""
간단한 추론 테스트 스크립트

Usage:
    python test_inference.py --model_path outputs/sft-bccard/final_model
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/sft-bccard/final_model",
        help="Trained model path"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizers/",
        help="Tokenizer path"
    )
    args = parser.parse_args()

    print("="*80)
    print("MOAI-LLM Inference Test")
    print("="*80)

    # 모델 로드
    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # GPU 사용 및 bf16 로드 (메모리 절약 + 속도 향상)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        model = model.to(device)
    
    model.eval()

    # 테스트 프롬프트
    test_prompts = [
        "What is artificial intelligence?",
        "Python으로 'Hello World'를 출력하는 방법은?",
        "신용카드 연회비는 어떻게 되나요?",
    ]

    print("="*80)
    print("Testing...")
    print("="*80)

    for prompt in test_prompts:
        print(f"\n[User]: {prompt}")

        # 포맷팅
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # 토큰화
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

        # 생성 (inference_mode가 no_grad보다 빠름)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 디코딩
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Assistant 응답만 추출
        try:
            response = response.split("<|im_start|>assistant")[-1]
            response = response.split("<|im_end|>")[0].strip()
        except:
            response = response

        print(f"[Assistant]: {response}")
        print("-"*80)

    print("\n✓ Test completed!")

if __name__ == "__main__":
    main()
