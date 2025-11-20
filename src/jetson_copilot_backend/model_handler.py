from llama_cpp import Llama
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", model_file="*q8_0.gguf", n_gpu_layers=-1):
        self.model_name = model_name
        self.model_file = model_file
        self.n_gpu_layers = n_gpu_layers
        self.model = None

    def load_model(self):
        logger.info(f"Loading model {self.model_name} ({self.model_file})...")
        try:
            self.model = Llama.from_pretrained(
                repo_id=self.model_name,
                filename=self.model_file,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=4096, # Adjust context window as needed
                verbose=True
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_completion(self, prompt, max_new_tokens=100):
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")

        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful code completion assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.1,
        )

        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    handler = ModelHandler()
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
