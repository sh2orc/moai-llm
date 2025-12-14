"""
Inference utilities for MOAI-LLM.

Provides easy-to-use inference pipeline for text generation.
"""

import torch
from typing import List, Optional, Union
from transformers import AutoTokenizer, GenerationConfig

from moai_llm.modeling.model import MoaiForCausalLM


class MoaiInferencePipeline:
    """
    Inference pipeline for MOAI-LLM.

    Simple interface for text generation with pre-trained models.

    Args:
        model_path: Path to pre-trained model checkpoint
        tokenizer_path: Path to tokenizer
        device: Device for computation ('cuda' or 'cpu')
        dtype: Data type for model weights (torch.float16, torch.bfloat16, torch.float32)

    Example:
        >>> pipeline = MoaiInferencePipeline("outputs/moai-3b/final_model", "tokenizers/moai_tokenizer")
        >>> text = pipeline.generate("Once upon a time", max_new_tokens=100)
        >>> print(text)
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.dtype = dtype

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = MoaiForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)

        self.model.eval()

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s).

        Args:
            prompt: Input prompt(s) (string or list of strings)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (vs greedy)
            num_return_sequences: Number of sequences to generate per prompt
            repetition_penalty: Penalty for repeating tokens

        Returns:
            Generated text(s)
        """
        # Handle single prompt vs batch
        single_prompt = isinstance(prompt, str)
        if single_prompt:
            prompt = [prompt]

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Return single string if single prompt
        if single_prompt and num_return_sequences == 1:
            return generated_texts[0]

        return generated_texts

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Chat interface with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
                Example: [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        # Format messages into prompt
        prompt = self._format_chat_prompt(messages)

        # Generate response
        response = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

        # Extract only the new response (remove prompt)
        response = response[len(prompt):].strip()

        return response

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into a prompt.

        Uses simple format with special tokens.
        """
        prompt = ""

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        # Add assistant prefix for generation
        prompt += "<|im_start|>assistant\n"

        return prompt

    def __call__(self, *args, **kwargs):
        """Shortcut for generate()."""
        return self.generate(*args, **kwargs)
