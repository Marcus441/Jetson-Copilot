import logging
from .model_handler import ModelHandler

logger = logging.getLogger(__name__)

class CompletionService:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", model_file="*q8_0.gguf", device="cuda"):
        # Map device to n_gpu_layers
        # -1 means all layers on GPU, 0 means all on CPU
        n_gpu_layers = -1 if device == "cuda" else 0
        self.model_handler = ModelHandler(model_name=model_name, model_file=model_file, n_gpu_layers=n_gpu_layers)
        logger.info(f"CompletionService initialized with model: {model_name} on device: {device}")

    def load_model(self):
        logger.info("Loading model for CompletionService...")
        try:
            self.model_handler.load_model()
            logger.info("Model loaded successfully for CompletionService.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_completion(self, prompt, max_new_tokens=100):
        logger.info(f"Generating completion for prompt (truncated): {prompt[:50]}...")
        try:
            completion = self.model_handler.generate_completion(prompt, max_new_tokens)
            logger.info("Completion generated successfully.")
            return completion
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise

    def get_chat_completion(self, messages, max_tokens=512, temperature=0.7):
        logger.info(f"Generating chat completion for {len(messages)} messages...")
        try:
            completion = self.model_handler.create_chat_completion(messages, max_tokens, temperature)
            logger.info("Chat completion generated successfully.")
            return completion
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            raise

    def get_ollama_completion(self, prompt, max_tokens=512, temperature=0.7, stop=None):
        logger.info(f"Generating Ollama completion for prompt (truncated): {prompt[:50]}...")
        try:
            completion = self.model_handler.create_completion(prompt, max_tokens, temperature, stop)
            logger.info("Ollama completion generated successfully.")
            return completion
        except Exception as e:
            logger.error(f"Error generating Ollama completion: {e}")
            raise

if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    # Consider using "cpu" for initial testing if a GPU is not readily available or for debugging purposes
    # For Jetson Orin Nano, you would typically use "cuda" after optimization
    service = CompletionService(device="cpu")
    service.load_model()

    test_prompt = "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n\n# Write a python function to check if a number is prime"

    print("\nGenerating code completion...")
    completion = service.get_completion(test_prompt, max_new_tokens=150)
    print("Completion:")
    print(completion)

