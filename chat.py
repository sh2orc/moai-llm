"""
대화형 채팅 인터페이스

Usage:
    python chat.py --model_path outputs/sft-bccard/final_model
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
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling"
    )
    args = parser.parse_args()

    print("="*80)
    print("MOAI-LLM Chat Interface")
    print("="*80)

    # 모델 로드
    print(f"Loading model from: {args.model_path}")
    print(f"Loading tokenizer from: {args.tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval()

    # GPU 사용
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)

    print("="*80)
    print("Ready! Type 'exit' to quit.")
    print("="*80)

    # 대화 루프
    conversation_history = []

    while True:
        # 사용자 입력
        user_input = input("\n💬 You: ")

        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye! 👋")
            break

        if not user_input.strip():
            continue

        # 대화 히스토리에 추가
        conversation_history.append(f"<|im_start|>user\n{user_input}<|im_end|>")

        # 프롬프트 구성
        prompt = "\n".join(conversation_history) + "\n<|im_start|>assistant\n"

        # 토큰화
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 디코딩
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Assistant 응답만 추출
        try:
            response = full_response.split("<|im_start|>assistant")[-1]
            response = response.split("<|im_end|>")[0].strip()
        except:
            response = full_response

        # 대화 히스토리에 추가
        conversation_history.append(f"<|im_start|>assistant\n{response}<|im_end|>")

        # 출력
        print(f"\n🤖 Assistant: {response}")

        # 대화 히스토리 길이 제한 (메모리 관리)
        if len(conversation_history) > 10:  # 최근 10턴만 유지
            conversation_history = conversation_history[-10:]

if __name__ == "__main__":
    main()
