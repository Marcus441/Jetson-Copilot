# src/jetson_project/model_handler.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelHandler:
    def __init__(
        self, model_name="Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int8", device="cuda"
    ):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(f"Loading model {self.model_name} to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=(torch.float16 if self.device == "cuda" else torch.float32),
            # Use float16 for GPU for performance, bfloat16 is not always supported on Jetson
            # For further optimization on Jetson, consider:
            # 1. Quantization: Post-training quantization (PTQ) or Quantization Aware Training (QAT)
            # 2. TensorRT: Convert the model to ONNX and then to TensorRT engine for maximum inference speed.
            #    This might require specific Jetson-compatible builds of PyTorch and TensorRT.
            device_map=self.device,
        )
        self.model.eval()
        print("Model loaded successfully.")

    def generate_completion(self, prompt, max_new_tokens=100):
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful code completion assistant.",
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,  # Ensure pad_token_id is set
        )

        # Decode only the newly generated tokens
        input_length = model_inputs.input_ids.shape[1]
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_length:], skip_special_tokens=True
        )
        return generated_text


if __name__ == "__main__":
    # Example usage
    handler = ModelHandler(device="cpu")  # Use "cuda" if a GPU is available
    handler.load_model()

    test_prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\n# write a python function to calculate the factorial of a number"

    print("\nGenerating code completion...")
    completion = handler.generate_completion(test_prompt, max_new_tokens=150)
    print("Completion:")
    print(completion)

    test_prompt_2 = "write a quick sort algorithm in python."
    print("\nGenerating another code completion...")
    completion_2 = handler.generate_completion(test_prompt_2, max_new_tokens=200)
    print("Completion 2:")
    print(completion_2)
