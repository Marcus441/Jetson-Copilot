# main.py
import logging
from .model_handler import ModelHandler

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize the ModelHandler
    # You can specify model_name, model_file, and n_gpu_layers here if needed
    # Default is n_gpu_layers=-1 (all on GPU)
    handler = ModelHandler()

    try:
        handler.load_model()

        prompt = "write a quick sort algorithm."

        # Generate completion using the ModelHandler
        response = handler.generate_completion(prompt, max_new_tokens=512)

        print("Response:")
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
